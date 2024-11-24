import os
import pygame
import numpy as np
import random
import agentpy as ap
import requests_simulador as rs

# Configuración inicial
WIDTH, HEIGHT = 800, 600  # Ampliamos el ancho para dejar espacio a las gráficas
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, (WIDTH - 200) // GRID_SIZE  # Ajustamos las columnas para el campo
print(ROWS, COLS)
TRACTOR_COUNT = 4
TRACTOR_SPEED = 5

# Colores
COLOR_EMPTY = (255, 255, 255)
COLOR_HARVESTED = (200, 200, 200)
COLOR_READY = (100, 255, 100)
COLOR_TRACTOR = (255, 100, 100)
COLOR_CONTAINER = (150, 150, 255)
COLOR_UNLOADING = (255, 255, 0)
COLOR_FUEL = (100, 100, 255)
COLOR_CARGO = (255, 165, 0)

# Inicialización de pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Cosecha con Gráficas en Tiempo Real")

# Clase para representar una parcela
class Parcel(ap.Agent):
    def setup(self):
        self.ready_to_harvest = True
        self.harvested = False
        self.reservada = False
        self.reservada_counter = 0
    
    def harvest(self):
        if self.ready_to_harvest:
            self.ready_to_harvest = False
            self.harvested = True

# Clase para el tractor/agente
class Tractor(ap.Agent):
    def setup(self, initial_position, id):
        self.id = id
        self.speed = TRACTOR_SPEED
        self.carga_max = 50
        self.carga_anterior = 0
        self.carga_actual = 0
        self.combustible_max = 1000
        self.combustible = self.combustible_max
        self.combustible_rate = 1
        self.position = np.array(initial_position, dtype=float)
        rs.send_coordinates_background(self.id, round(self.position[0]), round(self.position[1]))
        self.previous_position = self.position.copy()
        self.objetivo_actual = None
        self.descargando = False
        self.descarga_duracion = 0
        self.contador_descarga = 0
        self.contenedor = Container(self.position.copy())
        self.direccion = None  
        self.direccion_anterior = None
        self.q_table = np.zeros((ROWS, COLS))
        self.epsilon = 0.8 # Mantener en 1 para entrenar, bajar a 0.05 para usar Q-table entrenada
        self.alpha = 0.5 # Tasa de aprendizaje
        self.gamma = 0.75 # Factor de descuento
        self.direccion = None  
        self.cosechado_flag = False
        self.no_move_counter = 0
        self.lost_flag = False
        self.save_flag = False
        self.fuel_flag = False
        self.load_q_table()  # Cargar Q-table si existe
        self.siguiente_estado = None

    def mover(self, destino):
        if self.combustible > 0 and not self.descargando:
            direccion = destino - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > 0:
                self.direccion_anterior = self.direccion  # Guardar dirección anterior
                self.direccion = direccion / distancia  # Actualizar dirección actual
                direccion = (direccion / distancia) * self.speed
                self.position += direccion
                self.position[0] = np.clip(self.position[0], 0, (COLS - 0.5) * GRID_SIZE)
                self.position[1] = np.clip(self.position[1], 0, (ROWS - 0.5) * GRID_SIZE)
                self.combustible -= 0.13 * (np.linalg.norm(direccion) * self.combustible_rate)
                self.combustible_rate = 1

        if self.combustible <= 0:
            pass

    def cargar(self):
        if self.carga_actual < self.carga_max:
            self.carga_anterior = self.carga_actual
            self.carga_actual += 1
            self.cosechado_flag = True
            return True
        return False

    def descargar(self):
        self.descargando = True
        self.carga_actual = 0
        self.descarga_duracion = 30  # Duración de descarga
        self.contador_descarga = self.descarga_duracion
        self.contenedor.ir_al_silo_flag = True  # Enviar mensaje al contenedor para ir al silo

    def esperar(self):
        self.speed = 0

    def mover_a_contenedor(self):
        # Mueve el tractor hacia el contenedor para la descarga
        if not self.contenedor.ir_al_silo_flag:
            self.mover(self.contenedor.position)
        else:
            self.esperar()

    def seleccionar_accion(self, parcelas_disponibles):
        x = int(self.position[1] // GRID_SIZE)
        y = int(self.position[0] // GRID_SIZE)
        cargo = self.carga_actual
        combustible = int(self.combustible // 100)

        x = np.clip(x, 0, ROWS - 1)
        y = np.clip(y, 0, COLS - 1)
        cargo = np.clip(cargo, 0, self.carga_max)
        combustible = np.clip(combustible, 0, int(self.combustible_max / 100))

        if np.random.rand() < self.epsilon:
            accion = random.randint(0, 3)  # Acción aleatoria
        else:
            accion = np.argmax(self.q_table[x, y])  # Acción codiciosa

        direction_vectors = {
            0: np.array([-1, 0]),  # Arriba
            1: np.array([1, 0]),   # Abajo
            2: np.array([0, -1]),  # Izquierda
            3: np.array([0, 1])    # Derecha
        }

        if accion in direction_vectors:
            direction = direction_vectors[accion]
            next_x, next_y = x + direction[0], y + direction[1]

            next_x = np.clip(next_x, 0, ROWS - 1)
            next_y = np.clip(next_y, 0, COLS - 1)
            self.siguiente_estado = (next_x, next_y, cargo, combustible)

            if parcelas_disponibles:
                for parcela in parcelas_disponibles:
                    if parcela[0] == next_x and parcela[1] == next_y:
                        self.objetivo_actual = (parcela[0], parcela[1])
                        self.model.campo[parcela[0]][parcela[1]].reservada = True
                        break
            else:
                self.objetivo_actual = (next_x, next_y)

        return accion

    def step(self):
        parcelas_disponibles = self.model.obtener_parcelas_disponibles(self)
        
        # Verificar si el tractor no se ha movido desde el último paso
        if np.array_equal(self.position, self.previous_position):
            self.no_move_counter += 1
        else:
            self.no_move_counter = 0
        
        # Tomar una acción si no hay un objetivo específico
        if self.objetivo_actual is None:
            accion = self.seleccionar_accion(parcelas_disponibles)
            estado = (int(self.position[1] // GRID_SIZE), int(self.position[0] // GRID_SIZE), 
                      self.carga_actual, int(self.combustible // 100))
            recompensa = self.recompensa(accion)
            self.actualizar_q_valor(estado, accion, recompensa, self.siguiente_estado)
            
            if self.no_move_counter >= 20:
                self.forzar_mover_a_parcela_mas_cercana(parcelas_disponibles)
                self.no_move_counter = 0

        # Actualizar previous_position al final del paso
        self.previous_position = self.position.copy()

    def recompensa(self, accion):
        # Posición actual del tractor
        x = int(self.position[1] // GRID_SIZE)
        y = int(self.position[0] // GRID_SIZE)

        # Asegurarse de que los índices estén dentro de los límites
        x = np.clip(x, 0, ROWS - 1)
        y = np.clip(y, 0, COLS - 1)

        recompensa = 0

        # Recompensa por moverse a una parcela para cosechar
        if self.carga_anterior < self.carga_actual and self.cosechado_flag:
            self.cosechado_flag = False
            recompensa += 0.5  # Recompensa positiva por cosechar

        # Recompensa adicional por moverse a la parcela más cercana sin cambiar de dirección
        if self.direccion_anterior is not None and np.allclose(self.direccion, self.direccion_anterior):
            recompensa += 2  # Recompensa pequeña positiva

        # Penalización por inactividad (si no se ha movido en varios pasos)
        if self.no_move_counter >= 20:
            recompensa -= 5  

        # Penalización por pasar por una parcela ya cosechada
        if self.model.campo[x][y].harvested:
            recompensa -= 3  # Penalización por pasar por una parcela ya cosechada

        # Penalización por colisión
        if self.detectar_colision():
            recompensa -= 3 
        
        if self.objetivo_actual:
            distancia = np.sqrt((x - self.objetivo_actual[0])**2 + 
                            (y - self.objetivo_actual[1])**2)
            # Recompensa inversamente proporcional a la distancia
            recompensa += 2.0 / (distancia + 1) # Recompensa por moverse hacia el objetivo

        return recompensa

# Método para detectar colisión con otros tractores (simple ejemplo)
    def detectar_colision(self):
        for otro_tractor in self.model.tractores:
            if otro_tractor.id != self.id and np.array_equal(self.position, otro_tractor.position):
                return True
        return False
    

    def actualizar_q_valor(self, estado, accion, recompensa, siguiente_estado):
        x, y, cargo, combustible = estado

        # Asegurarse de que los índices estén dentro de los límites
        x = np.clip(x, 0, ROWS - 1)
        y = np.clip(y, 0, COLS - 1)
        cargo = np.clip(cargo, 0, self.carga_max)
        combustible = np.clip(combustible, 0, int(self.combustible_max / 100))
        

        # Obtener Q máximo para el siguiente estado
        Q_max = np.max(self.q_table[x, y])

        # Actualizar Q-valor
        self.q_table[x, y] += self.alpha * (recompensa + self.gamma * Q_max - self.q_table[x, y])

    def forzar_mover_a_parcela_mas_cercana(self, parcelas_disponibles):
        if parcelas_disponibles:
            parcelas_disponibles.sort(key=lambda parcela: parcela[2])  # Ordenar por distancia
            parcela_mas_cercana = parcelas_disponibles[0]
            self.objetivo_actual = (parcela_mas_cercana[0], parcela_mas_cercana[1])
            self.combustible_rate = 30
            self.model.campo[parcela_mas_cercana[0]][parcela_mas_cercana[1]].reservada = True
            if not self.lost_flag:
                self.lost_flag = True

    def save_q_table(self):
        filename = f"q_table_tractor_{self.id}.npy"
        np.save(filename, self.q_table)
        print(f"Q-table guardada en {filename}")

    def load_q_table(self):
        filename = f"q_table_tractor_{self.id}.npy"
        if os.path.exists(filename):
            self.q_table = np.load(filename)
            print(f"Q-table cargada desde {filename}")
        else:
            print(f"No se encontró Q-table para el tractor {self.id}, comenzando con una nueva")
            self.save_q_table()

# Clase para el contenedor
class Container:
    def __init__(self, initial_position):
        self.position = np.array(initial_position, dtype=float)
        self.color = COLOR_CONTAINER
        self.velocidad = TRACTOR_SPEED * 1.2
        self.ir_al_silo_flag = False  # Bandera para ir al silo

    def seguir_tractor(self, tractor_pos):
        # Mueve el contenedor para seguir al tractor con retraso
        if not self.ir_al_silo_flag:
            direccion = tractor_pos - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > GRID_SIZE * 2:  # Mantenerse a una distancia
                direccion = (direccion / distancia) * self.velocidad
                self.position += direccion


    def acercarse_tractor(self, tractor_pos):
        # Mueve el contenedor hacia el tractor
        direccion = tractor_pos - self.position
        distancia = np.linalg.norm(direccion)
        if distancia > 10:
            direccion = (direccion / distancia) * self.velocidad
            self.position += direccion
        if distancia < 10:
            self.ir_al_silo_flag = True

    def ir_al_silo(self, silo_pos):
        # Mueve el contenedor hacia el silo
        if self.ir_al_silo_flag:
            direccion = silo_pos - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > 10:
                direccion = (direccion / distancia) * self.velocidad
                self.position += direccion
            if distancia < 10:
                self.ir_al_silo_flag = False


# Clase para el modelo de simulación
class HarvestSimulation(ap.Model):
    def setup(self):
        self.campo = [[Parcel(self) for _ in range(COLS)] for _ in range(ROWS)]
        
        espaciado_x = (WIDTH - 180) // (TRACTOR_COUNT)
        posiciones_iniciales = [
            (WIDTH - 200 - i * espaciado_x, HEIGHT + GRID_SIZE) for i in range(TRACTOR_COUNT)
        ]
        
        self.tractores = [Tractor(self, initial_position=pos, id=i) for i, pos in enumerate(posiciones_iniciales)]
        
        margin_top = 20
        self.silo_position = (WIDTH - 180, margin_top + len(self.tractores) * 70 + 30)
    
    def obtener_parcelas_disponibles(self, tractor):
        parcelas_disponibles = []
        tractor_pos_grid = (int(tractor.position[1] // GRID_SIZE), int(tractor.position[0] // GRID_SIZE))
        
        for row in range(ROWS):
            for col in range(COLS):
                parcela = self.campo[row][col]
                if parcela.ready_to_harvest and not parcela.reservada:
                    distancia = np.linalg.norm(np.array(tractor_pos_grid) - np.array([row, col]))
                    parcelas_disponibles.append((row, col, distancia))
        
        return parcelas_disponibles

    def step(self):
        self.font = pygame.font.Font(None, 24)  # Define la fuente una vez
        screen.fill(COLOR_EMPTY)
        self.dibujar_campo()
        
        if self.all_parcels_harvested():
            print("All parcels have been harvested. Stopping simulation.")
            for tractor in self.tractores:
                tractor.save_q_table()
            #model.setup()
            return
        
        for idx, tractor in enumerate(self.tractores):
            if tractor.descargando:
                tractor.contador_descarga -= 1
                tractor.contenedor.color = COLOR_UNLOADING
                if tractor.contador_descarga <= 0:
                    tractor.descargando = False
                    tractor.contenedor.color = COLOR_CONTAINER
            else:
                if tractor.combustible <= 0:
                    tractor.contenedor.acercarse_tractor(tractor.position)
                    if np.linalg.norm(tractor.position - tractor.contenedor.position) < GRID_SIZE:
                        tractor.combustible = tractor.combustible_max
                        tractor.fuel_flag = True
                if tractor.carga_actual >= tractor.carga_max:
                    tractor.contenedor.acercarse_tractor(tractor.position)
                    if np.linalg.norm(tractor.position - tractor.contenedor.position) < GRID_SIZE:
                        tractor.descargar()
                else:
                    if tractor.objetivo_actual is None:
                        self.parcelas_disponibles = self.obtener_parcelas_disponibles(tractor)
                        tractor.step()
                    
                    elif tractor.objetivo_actual:
                        destino = np.array([
                            tractor.objetivo_actual[1] * GRID_SIZE + GRID_SIZE // 2, 
                            tractor.objetivo_actual[0] * GRID_SIZE + GRID_SIZE // 2
                        ])
                        tractor.mover(destino)
                        
                        if np.linalg.norm(destino - tractor.position) < tractor.speed:
                            if tractor.cargar():
                                self.campo[tractor.objetivo_actual[0]][tractor.objetivo_actual[1]].harvest()
                                rs.send_coordinates_background(tractor.id, round(tractor.position[0]), round(tractor.position[1]))
                            tractor.objetivo_actual = None
                
            if tractor.contenedor.ir_al_silo_flag:
                tractor.contenedor.ir_al_silo(self.silo_position)
            else:
                tractor.contenedor.seguir_tractor(tractor.position)
        
        self.dibujar_tractores()
        self.dibujar_graficas()
        self.dibujar_silo()
        pygame.display.flip()

    def dibujar_campo(self):
        for row in range(ROWS):
            for col in range(COLS):
                x, y = col * GRID_SIZE, row * GRID_SIZE
                parcela = self.campo[row][col]
                color = COLOR_HARVESTED if parcela.harvested else COLOR_READY if parcela.ready_to_harvest else COLOR_EMPTY
                if parcela.reservada:
                    parcela.reservada_counter += 1
                    color = (255, 0, 0)
                if parcela.reservada_counter > 8:
                    parcela.reservada = False
                    parcela.reservada_counter = 0
                pygame.draw.rect(screen, color, (x, y, GRID_SIZE, GRID_SIZE))
    
    def dibujar_tractores(self):
        for idx, tractor in enumerate(self.tractores):
            # Dibujar el contenedor y el tractor
            pygame.draw.circle(screen, tractor.contenedor.color, tractor.contenedor.position.astype(int), GRID_SIZE // 3)
            pygame.draw.circle(screen, COLOR_TRACTOR, tractor.position.astype(int), GRID_SIZE // 3)
            
            # Etiqueta del tractor
            label = self.font.render(f"Tractor {idx + 1}", True, (0, 0, 0))
            screen.blit(label, (tractor.position[0] - 15, tractor.position[1] - 30))  # Ajusta la posición del label si es necesario
    
    def dibujar_graficas(self):
        bar_width = 150
        bar_height = 20
        margin_top = 20
        for idx, tractor in enumerate(self.tractores):
            # Dibujar barra de combustible
            fuel_ratio = tractor.combustible / tractor.combustible_max
            pygame.draw.rect(screen, COLOR_FUEL, (WIDTH - 180, margin_top + idx * 70, int(bar_width * fuel_ratio), bar_height))
            pygame.draw.rect(screen, (0, 0, 0), (WIDTH - 180, margin_top + idx * 70, bar_width, bar_height), 2)

            # Dibujar barra de carga
            cargo_ratio = tractor.carga_actual / tractor.carga_max
            pygame.draw.rect(screen, COLOR_CARGO, (WIDTH - 180, margin_top + idx * 70 + 30, int(bar_width * cargo_ratio), bar_height))
            pygame.draw.rect(screen, (0, 0, 0), (WIDTH - 180, margin_top + idx * 70 + 30, bar_width, bar_height), 2)

            # Etiqueta del tractor
            font = pygame.font.Font(None, 24)
            text = font.render(f"Tractor {idx + 1}", True, (0, 0, 0))
            screen.blit(text, (WIDTH - 180, margin_top + idx * 70 - 20))

    def dibujar_silo(self):
        silo_width = 150
        silo_height = 150
        font = pygame.font.Font(None, 24)
        pygame.draw.rect(screen, (105, 105, 105), (self.silo_position[0], self.silo_position[1], silo_width, silo_height))
        label = font.render("Silo", True, (0, 0, 0))
        label_rect = label.get_rect(center=(self.silo_position[0] + silo_width // 2, self.silo_position[1] - 20))
        screen.blit(label, label_rect)

    def all_parcels_harvested(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.campo[row][col].ready_to_harvest:
                    return False
        return True

rs.check_connection_background(TRACTOR_COUNT)
# Ejecutar simulación
model = HarvestSimulation()
model.setup()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    model.step()
    #pygame.time.delay(5)

pygame.quit()
