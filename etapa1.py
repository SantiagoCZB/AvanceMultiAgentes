import pygame
import numpy as np
import random
import agentpy as ap

# Configuración inicial
WIDTH, HEIGHT = 800, 600  # Ampliamos el ancho para dejar espacio a las gráficas
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, (WIDTH - 200) // GRID_SIZE  # Ajustamos las columnas para el campo
TRACTOR_COUNT = 2
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
    
    def harvest(self):
        if self.ready_to_harvest:
            self.ready_to_harvest = False
            self.harvested = True
            self.reservada = False

# Clase para el tractor/agente
class Tractor(ap.Agent):
    def setup(self, initial_position):
        self.speed = TRACTOR_SPEED
        self.carga_max = 50
        self.carga_actual = 0
        self.combustible_max = 1000
        self.combustible = self.combustible_max
        self.position = np.array(initial_position, dtype=float)
        self.objetivo_actual = None
        self.descargando = False
        self.descarga_duracion = 0
        self.contador_descarga = 0
        self.contenedor = Container(self.position.copy())
    
    def mover(self, destino):
        if self.combustible > 0 and not self.descargando:
            direccion = destino - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > 0:
                direccion = (direccion / distancia) * self.speed
                self.position += direccion
                self.combustible -= 0.05 * np.linalg.norm(direccion)

    def cargar(self):
        if self.carga_actual < self.carga_max:
            self.carga_actual += 1
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

    def ir_al_silo(self, silo_pos):
        # Mueve el contenedor hacia el silo
        if self.ir_al_silo_flag:
            direccion = silo_pos - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > 10:
                print("Moviendo contenedor al silo")
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
            (WIDTH - 200 - i * espaciado_x, HEIGHT - GRID_SIZE // 2) for i in range(TRACTOR_COUNT)
        ]
        
        self.tractores = [Tractor(self, initial_position=pos) for pos in posiciones_iniciales]
        
        margin_top = 20
        self.silo_position = (WIDTH - 180, margin_top + len(self.tractores) * 70 + 30)
    
    def obtener_parcela_prioritaria(self, tractor):
        min_dist = float('inf')
        objetivo = None
        tractor_pos_grid = (int(tractor.position[1] // GRID_SIZE), int(tractor.position[0] // GRID_SIZE))
        
        for row in range(ROWS):
            for col in range(COLS):
                parcela = self.campo[row][col]
                if parcela.ready_to_harvest and not parcela.reservada:
                    distancia = np.linalg.norm(np.array(tractor_pos_grid) - np.array([row, col]))
                    if distancia < min_dist:
                        min_dist = distancia
                        objetivo = (row, col)
        
        if objetivo:
            self.campo[objetivo[0]][objetivo[1]].reservada = True
        return objetivo

    def step(self):
        self.font = pygame.font.Font(None, 24)  # Define la fuente una vez
        screen.fill(COLOR_EMPTY)
        self.dibujar_campo()
        
        for idx, tractor in enumerate(self.tractores):
            if tractor.descargando:
                tractor.contador_descarga -= 1
                tractor.contenedor.color = COLOR_UNLOADING
                if tractor.contador_descarga <= 0:
                    tractor.descargando = False
                    tractor.contenedor.color = COLOR_CONTAINER
            else:
                if tractor.carga_actual >= tractor.carga_max:
                    tractor.mover_a_contenedor()
                    if np.linalg.norm(tractor.position - tractor.contenedor.position) < GRID_SIZE:
                        tractor.descargar()
                else:
                    if tractor.objetivo_actual is None or self.campo[tractor.objetivo_actual[0]][tractor.objetivo_actual[1]].harvested:
                        tractor.objetivo_actual = self.obtener_parcela_prioritaria(tractor)
                    
                    if tractor.objetivo_actual:
                        destino = np.array([
                            tractor.objetivo_actual[1] * GRID_SIZE + GRID_SIZE // 2, 
                            tractor.objetivo_actual[0] * GRID_SIZE + GRID_SIZE // 2
                        ])
                        tractor.mover(destino)
                        
                        if np.linalg.norm(destino - tractor.position) < tractor.speed:
                            if tractor.cargar():
                                self.campo[tractor.objetivo_actual[0]][tractor.objetivo_actual[1]].harvest()
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

# Ejecutar simulación
model = HarvestSimulation()
model.setup()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    model.step()
    pygame.time.delay(50)

pygame.quit()
