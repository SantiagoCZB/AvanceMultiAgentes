import numpy as np
import pygame
import random
import agentpy as ap
from heapq import heappush, heappop
from enum import Enum

# Configuración inicial
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, (WIDTH - 200) // GRID_SIZE
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

# Direcciones de movimiento (arriba, derecha, abajo, izquierda)
class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

class PathNode:
    def __init__(self, position, g_cost, h_cost, parent=None, direction=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
        self.direction = direction

    def __lt__(self, other):
        return self.f_cost < other.f_cost

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
        self.path = []
        # Inicializar con dirección hacia arriba
        self.current_direction = Direction.UP
        self.previous_direction = Direction.UP
    
    def calcular_costo_giro(self, current_dir, new_dir):
        if current_dir is None:
            return 0
        # Penalizar más severamente los giros
        if current_dir != new_dir:
            # Penalizar más los giros de 180 grados
            if (current_dir == Direction.UP and new_dir == Direction.DOWN) or \
               (current_dir == Direction.DOWN and new_dir == Direction.UP) or \
               (current_dir == Direction.LEFT and new_dir == Direction.RIGHT) or \
               (current_dir == Direction.RIGHT and new_dir == Direction.LEFT):
                return 10
            return 5  # Penalización para giros de 90 grados
        return 0  # Sin costo para movimiento recto
    
    def encontrar_camino(self, start_grid, goal_grid, campo, otros_tractores):
        open_set = []
        closed_set = set()
        
        # Crear nodo inicial con la dirección actual
        start_node = PathNode(
            start_grid,
            0,
            self.manhattan_distance(start_grid, goal_grid),
            None,
            self.current_direction
        )
        heappush(open_set, start_node)
        
        while open_set:
            current = heappop(open_set)
            
            if current.position == goal_grid:
                self.previous_direction = self.current_direction
                return self.reconstruir_camino(current)
            
            if current.position in closed_set:
                continue
                
            closed_set.add(current.position)
            
            # Priorizar la dirección actual primero
            direcciones = list(Direction)
            if self.current_direction in direcciones:
                direcciones.remove(self.current_direction)
                direcciones.insert(0, self.current_direction)
            
            for direction in direcciones:
                dx, dy = direction.value
                new_pos = (current.position[0] + dx, current.position[1] + dy)
                
                if not (0 <= new_pos[0] < COLS and 0 <= new_pos[1] < ROWS):
                    continue
                
                if self.hay_colision(new_pos, otros_tractores):
                    continue
                
                giro_costo = self.calcular_costo_giro(current.direction, direction)
                g_cost = current.g_cost + 1 + giro_costo
                
                # Modificar la heurística para favorecer movimientos en la misma dirección
                h_cost = self.manhattan_distance(new_pos, goal_grid)
                if direction == self.current_direction:
                    h_cost *= 0.8  # Reducir el costo para favorecer movimiento recto
                
                neighbor = PathNode(
                    new_pos,
                    g_cost,
                    h_cost,
                    current,
                    direction
                )
                
                heappush(open_set, neighbor)
        
        return None

    def hay_colision(self, pos, otros_tractores):
        grid_pos = np.array(pos) * GRID_SIZE + GRID_SIZE // 2
        for otro in otros_tractores:
            if otro != self:
                dist = np.linalg.norm(otro.position - grid_pos)
                if dist < GRID_SIZE * 2:  # Margen de seguridad
                    return True
        return False

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reconstruir_camino(self, node):
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Invertir el camino para ir desde el inicio

    def mover(self, destino):
        if self.combustible > 0 and not self.descargando:
            current_grid = (
                int(self.position[0] // GRID_SIZE),
                int(self.position[1] // GRID_SIZE)
            )
            goal_grid = (
                int(destino[0] // GRID_SIZE),
                int(destino[1] // GRID_SIZE)
            )
            
            if not self.path:
                self.path = self.encontrar_camino(
                    current_grid,
                    goal_grid,
                    self.model.campo,
                    self.model.tractores
                )
            
            if self.path:
                next_grid = self.path[0]
                next_point = np.array([
                    next_grid[0] * GRID_SIZE + GRID_SIZE // 2,
                    next_grid[1] * GRID_SIZE + GRID_SIZE // 2
                ])
                
                direccion = next_point - self.position
                distancia = np.linalg.norm(direccion)
                
                if distancia > 0:
                    direccion = (direccion / distancia) * self.speed
                    self.position += direccion
                    self.combustible -= 0.05 * np.linalg.norm(direccion)
                
                if distancia < self.speed:
                    self.path.pop(0)
                    if self.path:
                        self.actualizar_direccion()

    def actualizar_direccion(self):
        if len(self.path) >= 2:
            current = np.array(self.path[0])
            next_pos = np.array(self.path[1])
            diff = next_pos - current
            
            if diff[0] > 0:
                self.current_direction = Direction.RIGHT
            elif diff[0] < 0:
                self.current_direction = Direction.LEFT
            elif diff[1] > 0:
                self.current_direction = Direction.DOWN
            elif diff[1] < 0:
                self.current_direction = Direction.UP

    def cargar(self):
        if self.carga_actual < self.carga_max:
            self.carga_actual += 1
            return True
        return False

    def descargar(self):
        self.descargando = True
        self.carga_actual = 0
        self.descarga_duracion = 30
        self.contador_descarga = self.descarga_duracion
        self.contenedor.ir_al_silo_flag = True

    def esperar(self):
        self.speed = 0

    def mover_a_contenedor(self):
        # En vez de moverse hacia el contenedor, esperar a que llegue
        if np.linalg.norm(self.position - self.contenedor.position) > GRID_SIZE * 2:
            self.esperar()
        else:
            self.descargar()

class Container:
    def __init__(self, initial_position):
        self.position = np.array(initial_position, dtype=float)
        self.color = COLOR_CONTAINER
        self.velocidad = TRACTOR_SPEED * 1.4
        self.ir_al_silo_flag = False

    def seguir_tractor(self, tractor_pos):
        if not self.ir_al_silo_flag:
            direccion = tractor_pos - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > GRID_SIZE * 2:
                direccion = (direccion / distancia) * self.velocidad
                self.position += direccion

    def ir_al_silo(self, silo_pos):
        if self.ir_al_silo_flag:
            direccion = silo_pos - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > 10:
                direccion = (direccion / distancia) * self.velocidad
                self.position += direccion
            if distancia < 10:
                self.ir_al_silo_flag = False

class HarvestSimulation(ap.Model):
    
    def setup(self):
        self.campo = [[Parcel(self) for _ in range(COLS)] for _ in range(ROWS)]
        
        espaciado_x = (WIDTH - 180) // (TRACTOR_COUNT)
        posiciones_iniciales = [
            (WIDTH - 200 - i * espaciado_x, HEIGHT - GRID_SIZE // 2) for i in range(TRACTOR_COUNT)
        ]
        
        self.tractores = [Tractor(self, initial_position=pos) for pos in posiciones_iniciales]
        
        # Establecer dirección inicial para cada tractor
        for tractor in self.tractores:
            tractor.current_direction = Direction.UP
            tractor.previous_direction = Direction.UP
        
        margin_top = 20
        self.silo_position = (WIDTH - 180, margin_top + len(self.tractores) * 70 + 30)
    
    def obtener_parcela_prioritaria(self, tractor):
        min_cost = float('inf')
        objetivo = None
        tractor_pos_grid = (
            int(tractor.position[0] // GRID_SIZE),
            int(tractor.position[1] // GRID_SIZE)
        )
        
        for row in range(ROWS):
            for col in range(COLS):
                parcela = self.campo[row][col]
                if parcela.ready_to_harvest and not parcela.reservada:
                    # Usar Manhattan distance
                    distancia = abs(tractor_pos_grid[0] - col) + abs(tractor_pos_grid[1] - row)
                    
                    # Factor de prioridad para movimiento vertical
                    if tractor.current_direction in [Direction.UP, Direction.DOWN]:
                        if col == tractor_pos_grid[0]:  # Misma columna
                            distancia *= 0.5  # Reducir significativamente el costo
                    
                    # Evitar cambios bruscos de dirección
                    if tractor.previous_direction:
                        if tractor.previous_direction in [Direction.UP, Direction.DOWN]:
                            if col != tractor_pos_grid[0]:  # Diferente columna
                                distancia *= 2  # Aumentar el costo
                    
                    if distancia < min_cost:
                        min_cost = distancia
                        objetivo = (row, col)
        
        if objetivo:
            self.campo[objetivo[0]][objetivo[1]].reservada = True
        return objetivo

    def step(self):
        self.font = pygame.font.Font(None, 24)
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
                        tractor.path = []  # Resetear el camino cuando hay nuevo objetivo
                    
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
                            tractor.path = []
                
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
            # Dibujar el camino planeado para cada tractor
            if tractor.path:
                for i in range(len(tractor.path) - 1):
                    start_pos = np.array([
                        tractor.path[i][0] * GRID_SIZE + GRID_SIZE // 2,
                        tractor.path[i][1] * GRID_SIZE + GRID_SIZE // 2
                    ])
                    end_pos = np.array([
                        tractor.path[i + 1][0] * GRID_SIZE + GRID_SIZE // 2,
                        tractor.path[i + 1][1] * GRID_SIZE + GRID_SIZE // 2
                    ])
                    pygame.draw.line(screen, (0, 0, 255), start_pos, end_pos, 2)

            # Dibujar el contenedor y el tractor
            pygame.draw.circle(screen, tractor.contenedor.color, tractor.contenedor.position.astype(int), GRID_SIZE // 3)
            pygame.draw.circle(screen, COLOR_TRACTOR, tractor.position.astype(int), GRID_SIZE // 3)
            
            # Dibujar la dirección actual del tractor
            if tractor.current_direction:
                direction_point = tractor.position + np.array(tractor.current_direction.value) * GRID_SIZE
                pygame.draw.line(screen, (255, 0, 0), 
                               tractor.position.astype(int),
                               direction_point.astype(int), 2)
            
            # Etiqueta del tractor
            label = self.font.render(f"Tractor {idx + 1}", True, (0, 0, 0))
            screen.blit(label, (tractor.position[0] - 15, tractor.position[1] - 30))
    
    def dibujar_graficas(self):
        bar_width = 150
        bar_height = 20
        margin_top = 20
        for idx, tractor in enumerate(self.tractores):
            # Dibujar barra de combustible
            fuel_ratio = tractor.combustible / tractor.combustible_max
            pygame.draw.rect(screen, COLOR_FUEL, 
                           (WIDTH - 180, margin_top + idx * 70, 
                            int(bar_width * fuel_ratio), bar_height))
            pygame.draw.rect(screen, (0, 0, 0), 
                           (WIDTH - 180, margin_top + idx * 70, 
                            bar_width, bar_height), 2)

            # Dibujar barra de carga
            cargo_ratio = tractor.carga_actual / tractor.carga_max
            pygame.draw.rect(screen, COLOR_CARGO, 
                           (WIDTH - 180, margin_top + idx * 70 + 30, 
                            int(bar_width * cargo_ratio), bar_height))
            pygame.draw.rect(screen, (0, 0, 0), 
                           (WIDTH - 180, margin_top + idx * 70 + 30, 
                            bar_width, bar_height), 2)

            # Etiqueta del tractor
            text = self.font.render(f"Tractor {idx + 1}", True, (0, 0, 0))
            screen.blit(text, (WIDTH - 180, margin_top + idx * 70 - 20))

            # Mostrar dirección actual
            if tractor.current_direction:
                dir_text = f"Dir: {tractor.current_direction.name}"
                dir_label = self.font.render(dir_text, True, (0, 0, 0))
                screen.blit(dir_label, 
                           (WIDTH - 80, margin_top + idx * 70 + 5))

    def dibujar_silo(self):
        silo_width = 150
        silo_height = 150
        pygame.draw.rect(screen, (105, 105, 105), 
                        (self.silo_position[0], self.silo_position[1], 
                         silo_width, silo_height))
        label = self.font.render("Silo", True, (0, 0, 0))
        label_rect = label.get_rect(
            center=(self.silo_position[0] + silo_width // 2, 
                   self.silo_position[1] - 20)
        )
        screen.blit(label, label_rect)

# Ejecutar simulación
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Cosecha con A* y Movimiento Realista")

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