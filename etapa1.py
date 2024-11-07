import pygame
import numpy as np
import random
import agentpy as ap

# Configuración inicial
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE
TRACTOR_COUNT = 4
TRACTOR_SPEED = 5

# Colores
COLOR_EMPTY = (255, 255, 255)
COLOR_HARVESTED = (200, 200, 200)
COLOR_READY = (100, 255, 100)
COLOR_TRACTOR = (255, 100, 100)

# Inicialización de pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Cosecha con AgentPy")

# Clase para representar una parcela como un agente de AgentPy
class Parcel(ap.Agent):
    def setup(self):
        self.ready_to_harvest = True
        self.harvested = False
        self.reservada = False  # Nueva propiedad para indicar si está reservada por un tractor
        self.humidity = random.uniform(0.5, 1)
    
    def harvest(self):
        if self.ready_to_harvest:
            self.ready_to_harvest = False
            self.harvested = True
            self.reservada = False  # Liberar reserva al cosechar

# Clase para el tractor/agente
class Tractor(ap.Agent):
    def setup(self, initial_position):
        self.speed = TRACTOR_SPEED
        self.carga_max = 50000
        self.carga_actual = 0
        self.combustible = 100
        self.consumo_combustible = 0.00
        self.eficiencia = 0.9
        self.position = np.array(initial_position, dtype=float)
        self.objetivo_actual = None  # Guardar la parcela actual hacia la que se dirige

    def mover(self, destino):
        if self.combustible > 0:
            direccion = destino - self.position
            distancia = np.linalg.norm(direccion)
            if distancia > 0:
                direccion = (direccion / distancia) * self.speed
                self.position += direccion
                self.combustible -= self.consumo_combustible * np.linalg.norm(direccion)
                
    def cargar(self):
        if self.carga_actual < self.carga_max:
            self.carga_actual += 1
            return True
        return False

# Clase para el modelo de simulación
class HarvestSimulation(ap.Model):
    def setup(self):
        # Creación de campo y tractores
        self.campo = [[Parcel(self) for _ in range(COLS)] for _ in range(ROWS)]
        
        # Posicionar tractores en la parte más baja y distribuidos uniformemente
        espaciado_x = WIDTH // TRACTOR_COUNT
        posiciones_iniciales = [
            (i * espaciado_x + espaciado_x // 2, HEIGHT - GRID_SIZE // 2) for i in range(TRACTOR_COUNT)
        ]
        
        self.tractores = [Tractor(self, initial_position=pos) for pos in posiciones_iniciales]
    
    def obtener_parcela_prioritaria(self, tractor):
        min_dist = float('inf')
        objetivo = None
        tractor_pos_grid = (int(tractor.position[1] // GRID_SIZE), int(tractor.position[0] // GRID_SIZE))
        
        for row in range(ROWS):
            for col in range(COLS):
                parcela = self.campo[row][col]
                # Solo considerar parcelas que están listas y no están reservadas
                if parcela.ready_to_harvest and not parcela.reservada:
                    distancia = np.linalg.norm(np.array(tractor_pos_grid) - np.array([row, col]))
                    if distancia < min_dist:
                        min_dist = distancia
                        objetivo = (row, col)
        
        # Reservar la parcela seleccionada
        if objetivo:
            self.campo[objetivo[0]][objetivo[1]].reservada = True
        return objetivo

    def step(self):
        # Dibujado en Pygame
        screen.fill(COLOR_EMPTY)
        self.dibujar_campo()
        
        for tractor in self.tractores:
            # Solo buscar un nuevo objetivo si el tractor no tiene uno
            if tractor.objetivo_actual is None or self.campo[tractor.objetivo_actual[0]][tractor.objetivo_actual[1]].harvested:
                tractor.objetivo_actual = self.obtener_parcela_prioritaria(tractor)
            
            if tractor.objetivo_actual:
                destino = np.array([
                    tractor.objetivo_actual[1] * GRID_SIZE + GRID_SIZE // 2, 
                    tractor.objetivo_actual[0] * GRID_SIZE + GRID_SIZE // 2
                ])
                tractor.mover(destino)
                
                # Si el tractor alcanza la parcela, realiza la cosecha
                if np.linalg.norm(destino - tractor.position) < tractor.speed:
                    if tractor.cargar():
                        self.campo[tractor.objetivo_actual[0]][tractor.objetivo_actual[1]].harvest()
                    tractor.objetivo_actual = None  # Liberar el objetivo para buscar uno nuevo
        
        self.dibujar_tractores()
        pygame.display.flip()

    def dibujar_campo(self):
        for row in range(ROWS):
            for col in range(COLS):
                x, y = col * GRID_SIZE, row * GRID_SIZE
                parcela = self.campo[row][col]
                color = COLOR_HARVESTED if parcela.harvested else COLOR_READY if parcela.ready_to_harvest else COLOR_EMPTY
                pygame.draw.rect(screen, color, (x, y, GRID_SIZE, GRID_SIZE))
    
    def dibujar_tractores(self):
        for tractor in self.tractores:
            pygame.draw.circle(screen, COLOR_TRACTOR, tractor.position.astype(int), GRID_SIZE // 3)

# Ejecución de la simulación
model = HarvestSimulation()
model.setup()
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    model.step()
    clock.tick(30)

pygame.quit()
