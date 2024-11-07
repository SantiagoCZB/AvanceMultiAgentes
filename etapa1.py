import agentpy as ap
import pygame
import random
import heapq

# Parámetros de simulación
GRID_SIZE = 20  # Tamaño de la cuadrícula (20x20)
CELL_SIZE = 30  # Tamaño de cada celda en píxeles
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# Colores
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)  # Verde para el campo
BROWN = (139, 69, 19)  # Marrón para las parcelas cosechadas
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)      # Para los bordes de las celdas

# Configuración de pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Simulación de Tractores con Dijkstra")

# Definir el agente Tractor
class Tractor(ap.Agent):
    def setup(self, start_positions):
        # Inicializar variables de cada tractor
        self.fuel = 300  # Combustible inicial
        # Seleccionar una posición única de la fila inferior
        self.position = start_positions.pop()
        self.color = BLUE  # Color del tractor
        self.path = []  # Camino a seguir (listado de posiciones)
    
    def find_closest_target(self, field):
        """Encuentra la parcela sin cosechar más cercana usando Dijkstra"""
        start = tuple(self.position)
        queue = [(0, start)]
        distances = {start: 0}
        previous = {start: None}
        closest_target = None
        closest_distance = float('inf')

        while queue:
            dist, current = heapq.heappop(queue)
            x, y = current
            
            # Si encontramos una parcela sin cosechar, la tomamos como objetivo
            if field[y][x] == 0:
                if dist < closest_distance:
                    closest_target = current
                    closest_distance = dist
                    break

            # Revisar los vecinos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                    new_dist = dist + 1
                    if new_dist < distances.get(neighbor, float('inf')):
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(queue, (new_dist, neighbor))
        
        # Si se encontró una parcela sin cosechar, reconstruir el camino
        path = []
        if closest_target:
            while closest_target is not None:
                path.append(list(closest_target))
                closest_target = previous[closest_target]
            path.reverse()
        return path[1:]  # Omitir la posición inicial del camino

    def move(self, field, occupied_positions):
        """Mueve el tractor a lo largo de su camino y busca nuevo objetivo si necesario"""
        if self.fuel > 0 and not self.path:
            self.path = self.find_closest_target(field)  # Encuentra nuevo objetivo
        
        if self.fuel > 0 and self.path:
            next_position = self.path.pop(0)
            
            # Verificar que la siguiente posición no esté ocupada
            if tuple(next_position) not in occupied_positions:
                self.position = next_position
                self.fuel -= 1

# Definir el campo
class Campo(ap.Model):
    def setup(self):
        # Crear lista de posiciones de inicio en la fila inferior (y = GRID_SIZE - 1)
        start_positions = [[x, GRID_SIZE - 1] for x in range(GRID_SIZE)]
        random.shuffle(start_positions)  # Mezclar para asignar aleatoriamente

        # Crear lista de agentes (tractores) y asignar posiciones de inicio
        self.tractors = ap.AgentList(self, 5, Tractor, start_positions=start_positions[:5])
        self.field = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # Campo vacío
        
    def step(self):
        # Almacenar posiciones ocupadas para evitar colisiones
        occupied_positions = {tuple(tractor.position) for tractor in self.tractors}

        # Mover tractores y actualizar el estado del campo
        for tractor in self.tractors:
            tractor.move(self.field, occupied_positions)
            x, y = tractor.position
            self.field[y][x] = 1  # Marcar como cosechado

# Función para dibujar la simulación en pygame
def draw(simulation):
    screen.fill(WHITE)
    # Dibujar campo
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = BROWN if simulation.field[y][x] == 1 else GREEN  # Cambiar de verde a marrón
            pygame.draw.rect(screen, color, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLACK, pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
    
    # Dibujar tractores
    for tractor in simulation.tractors:
        x, y = tractor.position
        pygame.draw.circle(screen, tractor.color, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)

    pygame.display.flip()

# Inicializar simulación
simulation = Campo()
simulation.setup()

# Loop principal
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    simulation.step()
    draw(simulation)
    clock.tick(6)  # Velocidad de la simulación (10 FPS)

pygame.quit()
