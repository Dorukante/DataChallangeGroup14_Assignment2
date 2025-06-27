from time import perf_counter
import numpy as np
import pymunk
from pygame import gfxdraw
import pygame
import sys
from environment.environment_config import *
from typing import List, Tuple


class ContinuousGUI:
    # Enhanced color scheme from warehouse_gui
    COLOR_BG = (245, 245, 245)  # Light gray background
    COLOR_FLOOR = (235, 235, 235)  # Floor
    COLOR_AGENT = (0, 150, 0)  # Green robot
    COLOR_AGENT_ALT = (0, 100, 200)  # Alternative agent color (e.g., when carrying)
    COLOR_OBSTACLE = (100, 100, 100)  # Dark gray for obstacles
    COLOR_OBSTACLE_OUTLINE = (60, 60, 60)  # Obstacle borders
    COLOR_GOAL = (0, 0, 255)  # Blue goals
    COLOR_SENSOR = (255, 165, 0, 100)  # Orange sensors with transparency
    COLOR_SENSOR_HIT = (255, 0, 0, 150)  # Red when detecting
    COLOR_PATH_TRAIL = (100, 200, 100, 50)  # Green trail
    COLOR_COLLISION_INDICATOR = (128, 0, 128)
    COLOR_COLLISION_FLASH = (255, 0, 0)  # Red collision flash
    COLOR_TEXT = (40, 40, 40)  # Dark text
    COLOR_PANEL_BG = (250, 250, 250)  # Info panel background
    COLOR_SUCCESS = (0, 200, 0)  # Green for success
    COLOR_WARNING = (255, 165, 0)  # Orange for warnings
    COLOR_DANGER = (255, 0, 0)  # Red for danger

    # Enhanced info display
    INFO_NAME_MAP = [
        ("current_collision_count", "Total Collisions:"),
        ("elapsed_physics_time", "Simulation Time (s):"),
        ("distance_traveled", "Distance Traveled:"),
        ("agent_status", "Agent Status:"),
        ("fps", "FPS:"),
    ]

    def __init__(self, extents: tuple[int, int],
            window_size: tuple[int, int] = (1400, 800)):
        """Provides an enhanced GUI to show what is happening in the environment.

        Args:
            extents: Environment dimensions (width, height).
            window_size: The size of the pygame window. (width, height).
        """
        self.grid_size = extents

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Data Intelligence Challenge 2024 - Enhanced")
        self.clock = pygame.time.Clock()

        self.stats = self._reset_stats()

        self.grid_panel_size = (int(window_size[0] * 0.75), window_size[1])
        self.info_panel_rect = pygame.Rect(
            self.grid_panel_size[0],
            0,
            window_size[0] - self.grid_panel_size[0],
            window_size[1]
        )
        
        # Path tracking from warehouse_gui
        self.agent_path = []
        self.max_path_length = 500  # Keep last N positions
        self.path_draw_interval = 5  # Draw every Nth point
        
        # Visual effects from warehouse_gui
        self.collision_flash_timer = 0
        self.last_collision_count = 0
        
        # Visualization toggles
        self.show_sensors = True
        self.show_path = True
        self.show_labels = True
        
        # Initialize fonts
        pygame.font.init()
        self.title_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 20)  # Reduced from 24
        self.value_font = pygame.font.Font(None, 24)  # Reduced from 28
        self.small_font = pygame.font.Font(None, 20)

        self.paused = False
        self.paused_clicked = False
        self.step = False
        self.step_clicked = False

        # Find the smallest window dimension and max grid size to calculate the
        # grid scalar
        self.scalar = min(self.grid_panel_size)
        self.scalar /= max(self.grid_size) * 1.2

        # FPS timer
        self.last_render_time = perf_counter()
        self.last_10_fps = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.frame_count = 0
        
        # Additional tracking
        self.total_distance = 0.0
        self.last_agent_pos = None

        self._initial_render()

    def reset(self):
        """Called during the reset method of the environment."""
        self.stats = self._reset_stats()
        self.agent_path.clear()
        self.collision_flash_timer = 0
        self.total_distance = 0.0
        self.last_agent_pos = None
        self.last_collision_count = 0
        self._initial_render()

    @staticmethod
    def _reset_stats():
        return {
            "total_targets_reached": 0,
            "total_failed_move": 0,
            "current_collision_count": 0,
            "fps": "0.0",
            "total_steps": 0,
            "elapsed_physics_time": 0.0,
            "cumulative_reward": 0,
            "distance_traveled": 0.0,
            "agent_status": "Active"
        }

    def _initial_render(self):
        """Initial render of the environment with enhanced styling."""
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill(self.COLOR_BG)

        # Display the loading text with better styling
        text = self.title_font.render("Loading environment...", True, self.COLOR_TEXT)
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery

        background.blit(text, textpos)
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

    def _draw_button(self, surface: pygame.Surface, text: str,
            rect: pygame.Rect, color: tuple[int, int, int],
            text_color: tuple[int, int, int] = None, clicked: bool = False):
        """Draws an enhanced button on the given surface."""
        if text_color is None:
            text_color = self.COLOR_TEXT
            
        # Button shadow effect
        shadow_rect = rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        pygame.draw.rect(surface, (200, 200, 200), shadow_rect)
        
        # Main button
        button_color = (min(color[0] - 30, 255), min(color[1] - 30, 255), min(color[2] - 30, 255)) if clicked else color
        pygame.draw.rect(surface, button_color, rect)
        pygame.draw.rect(surface, self.COLOR_TEXT, rect, 2)

        font = self.label_font
        text_surface = font.render(text, True, text_color)
        textpos = text_surface.get_rect()
        textpos.centerx = rect.centerx
        textpos.centery = rect.centery
        surface.blit(text_surface, textpos)

    def _draw_path_trail(self, surface: pygame.Surface):
        """Draw agent's path trail (from warehouse_gui)."""
        if not self.show_path or len(self.agent_path) < 2:
            return
        
        # Draw path as connected lines with fading effect
        for i in range(0, len(self.agent_path) - 1, self.path_draw_interval):
            if i + self.path_draw_interval < len(self.agent_path):
                start = self.agent_path[i]
                end = self.agent_path[i + self.path_draw_interval]
                
                # Fade older points
                alpha = int(150 * (i / len(self.agent_path)))
                
                pygame.draw.line(surface, (100, 200, 100), start, end, 2)

    def _draw_enhanced_agent(self, surface: pygame.Surface, agent_pos: pymunk.Vec2d, 
                           agent_radius: float, agent_angle: float, has_goal: bool = False):
        """Draw agent with enhanced visualization (from warehouse_gui)."""
        x = int(agent_pos.x * self.scalar)
        y = int(agent_pos.y * self.scalar)
        radius = int(agent_radius * self.scalar)
        
        # Choose color based on state
        agent_color = self.COLOR_AGENT_ALT if has_goal else self.COLOR_AGENT
        
        # Draw agent shadow
        gfxdraw.filled_circle(surface, x + 2, y + 2, radius, (150, 150, 150))
        
        # Draw agent body
        gfxdraw.filled_circle(surface, x, y, radius, agent_color)
        gfxdraw.aacircle(surface, x, y, radius, agent_color)
        
        # Draw direction indicator
        end_x = x + int(np.cos(agent_angle) * radius * 0.8)
        end_y = y + int(np.sin(agent_angle) * radius * 0.8)
        pygame.draw.line(surface, (255, 255, 255), (x, y), (end_x, end_y), 3)
        
        # Collision flash effect (from warehouse_gui)
        if self.collision_flash_timer > 0:
            flash_radius = int(radius * 1.5)
            alpha = min(int(self.collision_flash_timer * 255), 255)
            # Draw expanding circle for flash effect
            pygame.draw.circle(surface, self.COLOR_COLLISION_FLASH, (x, y), flash_radius, 3)
            self.collision_flash_timer -= 0.05

    def _draw_obstacles_with_3d_effect(self, surface: pygame.Surface, obstacles: List[Tuple[pymunk.Body, pymunk.Shape]]):
        """Draw obstacles with 3D effect and labels (from warehouse_gui)."""
        for obstacle_body, obstacle_shape in obstacles:
            if isinstance(obstacle_shape, pymunk.Poly):
                B = obstacle_body.position
                vertices = obstacle_shape.get_vertices()
                
                # Scale vertices
                points = []
                for v in vertices:
                    x = int((v.x + B.x) * self.scalar)
                    y = int((v.y + B.y) * self.scalar)
                    points.append((x, y))
                
                # Draw shadow for 3D effect
                shadow_points = [(p[0] + 3, p[1] + 3) for p in points]
                pygame.draw.polygon(surface, (200, 200, 200), shadow_points)
                
                # Draw main obstacle
                pygame.draw.polygon(surface, self.COLOR_OBSTACLE, points)
                pygame.draw.polygon(surface, self.COLOR_OBSTACLE_OUTLINE, points, 2)
                
                # Add labels if enabled
                if self.show_labels:
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                    
                    # Simple shelf labeling based on size
                    width = max(p[0] for p in points) - min(p[0] for p in points)
                    height = max(p[1] for p in points) - min(p[1] for p in points)
                    
                    if 50 < width < 200 and 50 < height < 200:  # Likely a shelf
                        label = self.small_font.render("S", True, (255, 255, 255))
                        label_rect = label.get_rect(center=(int(center_x), int(center_y)))
                        surface.blit(label, label_rect)

    def _draw_enhanced_sensors(self, surface: pygame.Surface, sensors, scalar: float):
        """Draw sensors with enhanced visualization (from warehouse_gui)."""
        if not self.show_sensors:
            return
            
        for sensor in sensors:
            if not sensor.is_active:
                continue
                
            start_x = int(sensor.sensor_start[0] * scalar)
            start_y = int(sensor.sensor_start[1] * scalar)
            
            if sensor.name == "RaySensor":
                if sensor.sensed_object_type != 0.0:
                    end_x = int(sensor.sensed_object_position[0] * scalar)
                    end_y = int(sensor.sensed_object_position[1] * scalar)
                    
                    # RaySensor only detects obstacles and other agents, not goals
                    color = (255, 0, 0)  # Red for any detection
                    
                    pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)
                    gfxdraw.filled_circle(surface, end_x, end_y, 4, color)
                else:
                    end_x = int(sensor.sensor_end[0] * scalar)
                    end_y = int(sensor.sensor_end[1] * scalar)
                    pygame.draw.line(surface, (255, 165, 0), (start_x, start_y), (end_x, end_y), 1)
                    
            elif sensor.name == "RaySensorNoType":
                end_x = int(sensor.sensed_object_position[0] * scalar)
                end_y = int(sensor.sensed_object_position[1] * scalar)
                
                if sensor.sensed_object_position != sensor.sensor_end:
                    # Object detected
                    pygame.draw.line(surface, (255, 100, 100), (start_x, start_y), (end_x, end_y), 2)
                    gfxdraw.filled_circle(surface, end_x, end_y, 4, self.COLOR_COLLISION_INDICATOR)
                else:
                    # No detection - faint line
                    pygame.draw.line(surface, (255, 165, 0, 50), (start_x, start_y), (end_x, end_y), 1)

    def _draw_enhanced_info(self, surface) -> tuple[pygame.Rect, pygame.Rect]:
        """Draws the enhanced info panel on the surface."""
        # Panel background
        pygame.draw.rect(surface, self.COLOR_PANEL_BG, self.info_panel_rect)
        pygame.draw.rect(surface, self.COLOR_TEXT, self.info_panel_rect, 2)
        
        x_offset = self.grid_panel_size[0] + 20
        y_offset = 20

        # Title
        title = self.title_font.render("Environment Status", True, self.COLOR_TEXT)
        surface.blit(title, (x_offset, y_offset))
        y_offset += 50

        col_width = 180  # Reduced from 200
        row_height = 50  # Increased from 35 for two-line layout

        # Draw metrics
        for row, (key, name) in enumerate(self.INFO_NAME_MAP):
            y_pos = y_offset + (row * row_height)
            
            # Label
            label_text = self.label_font.render(name, True, self.COLOR_TEXT)
            surface.blit(label_text, (x_offset, y_pos))

            # Value with color coding
            value = str(self.stats[key])
            value_color = self.COLOR_TEXT
            
            # Color code collision count
            if key == "current_collision_count":
                if self.stats[key] == 0:
                    value_color = self.COLOR_SUCCESS
                elif self.stats[key] < 5:
                    value_color = self.COLOR_WARNING
                else:
                    value_color = self.COLOR_DANGER
            
            # Format distance
            if key == "distance_traveled":
                value = f"{self.stats[key]:.1f}"
            
            # Format agent status to fit and add color
            if key == "agent_status":
                if "PAUSED" in value:
                    value_color = self.COLOR_WARNING
                elif "Complete" in value:
                    value_color = self.COLOR_SUCCESS
                if len(value) > 15:
                    value = value[:15] + "..."
            
            value_text = self.value_font.render(value, True, value_color)
            # Draw value on next line for better spacing
            surface.blit(value_text, (x_offset + 20, y_pos + 18))

        # Control buttons with better spacing
        button_y = y_offset + (len(self.INFO_NAME_MAP) + 1) * row_height + 20
        
        # Pause/Resume button
        pause_rect = pygame.Rect(x_offset, button_y, 200, 50)
        self._draw_button(surface, "Resume" if self.paused else "Pause",
                         pause_rect, (255, 255, 255), clicked=self.paused_clicked)

        # Step button
        step_rect = pygame.Rect(x_offset, button_y + 60, 200, 50)
        self._draw_button(surface, "Step", step_rect, (255, 255, 255), clicked=self.step_clicked)
        
        # Toggle buttons
        toggle_y = button_y + 130
        button_width = 180
        button_height = 35
        button_spacing = 10
        
        toggles = [
            ("Sensors", self.show_sensors),
            ("Path", self.show_path),
            ("Labels", self.show_labels)
        ]
        
        for i, (text, state) in enumerate(toggles):
            toggle_rect = pygame.Rect(x_offset, toggle_y + i * (button_height + button_spacing),
                                    button_width, button_height)
            color = self.COLOR_SUCCESS if state else (200, 200, 200)
            self._draw_button(surface, f"Toggle {text}", toggle_rect, color)

        return pause_rect, step_rect

    def render(self, env, reward: float = 0, is_single_step: bool = False):
        """Enhanced render method with all warehouse_gui improvements."""
        info = env.info
        
        # Check if we should actually process this frame
        should_process = (not self.paused and not self.step) or is_single_step
        
        # Handle step mode
        if self.step and not is_single_step:
            self.step = False  # Reset step flag

        # Update collision flash
        if env.agent_collided_with_obstacle_count_after > self.last_collision_count:
            self.collision_flash_timer = 1.0
        self.last_collision_count = env.agent_collided_with_obstacle_count_after

        # Update path trail
        agent_pos = env.agent_body.position
        scaled_pos = (int(agent_pos.x * self.scalar), int(agent_pos.y * self.scalar))
        self.agent_path.append(scaled_pos)
        if len(self.agent_path) > self.max_path_length:
            self.agent_path.pop(0)

        # Update distance traveled
        if self.last_agent_pos is not None:
            distance = np.sqrt((agent_pos.x - self.last_agent_pos.x)**2 + 
                             (agent_pos.y - self.last_agent_pos.y)**2)
            self.total_distance += distance
            self.stats["distance_traveled"] = self.total_distance
        self.last_agent_pos = pymunk.Vec2d(agent_pos.x, agent_pos.y)

        # Update FPS
        self.frame_count += 1
        self.frame_count %= 10

        curr_time = perf_counter()
        fps = 1 / (curr_time - self.last_render_time)
        self.last_render_time = curr_time

        self.last_10_fps[self.frame_count] = fps
        self.stats["fps"] = f"{sum(self.last_10_fps) / 10:.1f}"
        self.stats["total_targets_reached"] += int(info["target_reached"])
        self.stats["elapsed_physics_time"] = env.world_stats["total_time"]
        self.stats["current_collision_count"] = env.world_stats["collision_count"]

        if (not self.paused and not self.step) or is_single_step:
            self.stats["total_steps"] += 1

        failed_move = 1 - int(info["agent_moved"])
        self.stats["total_failed_move"] += failed_move
        self.stats["cumulative_reward"] += reward

        # Update agent status
        if len(env.current_goals) == 0:
            self.stats["agent_status"] = "Complete"
        elif self.paused:
            self.stats["agent_status"] = "PAUSED"
        else:
            goals_left = len(env.current_goals)
            self.stats["agent_status"] = f"Active ({goals_left}g)"  # Shortened

        # Create main surface
        background = pygame.Surface(self.window.get_size()).convert()
        background.fill(self.COLOR_BG)

        # Draw floor with pause overlay if paused
        floor_rect = pygame.Rect(0, 0, self.grid_panel_size[0], self.grid_panel_size[1])
        pygame.draw.rect(background, self.COLOR_FLOOR, floor_rect)
        
        # Add pause overlay
        if self.paused:
            s = pygame.Surface((self.grid_panel_size[0], 50))
            s.set_alpha(200)
            s.fill(self.COLOR_WARNING)
            background.blit(s, (0, 0))
            pause_text = self.title_font.render("PAUSED", True, self.COLOR_TEXT)
            text_rect = pause_text.get_rect(center=(self.grid_panel_size[0]//2, 25))
            background.blit(pause_text, text_rect)

        # Draw in proper order for visual hierarchy
        self._draw_path_trail(background)
        
        # Draw goals with enhanced style
        for goal_obj in env.current_goals.values():
            goal = goal_obj.body.position
            x = int(goal.x * self.scalar)
            y = int(goal.y * self.scalar)
            radius = int(GOAL_RADIUS * self.scalar)
            
            # Goal shadow
            gfxdraw.filled_circle(background, x + 2, y + 2, radius, (150, 150, 150))
            # Goal body
            gfxdraw.filled_circle(background, x, y, radius, self.COLOR_GOAL)
            gfxdraw.aacircle(background, x, y, radius, self.COLOR_GOAL)

        # Draw obstacles with 3D effect
        self._draw_obstacles_with_3d_effect(background, env.obstacles)

        # Draw sensors
        self._draw_enhanced_sensors(background, env.agent_state.sensors, self.scalar)

        # Draw agent
        self._draw_enhanced_agent(background, env.agent_body.position, 
                                AGENT_RADIUS, env.agent_body.angle,
                                len(env.current_goals) < len(env.initial_goal_positions))

        # Draw info panel
        pause_rect, step_rect = self._draw_enhanced_info(background)

        # Blit the surface onto the window
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

        # Parse events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                
                # Check pause button
                if pause_rect.collidepoint(event.pos):
                    self.paused_clicked = True
                elif step_rect.collidepoint(event.pos):
                    self.step_clicked = True
                    
                # Check toggle buttons
                x_offset = self.grid_panel_size[0] + 20
                toggle_y = pause_rect.bottom + 80
                button_width = 180
                button_height = 35
                button_spacing = 10
                
                # Toggle sensors
                if (x_offset <= mouse_x <= x_offset + button_width and
                    toggle_y <= mouse_y <= toggle_y + button_height):
                    self.show_sensors = not self.show_sensors
                # Toggle path
                elif (x_offset <= mouse_x <= x_offset + button_width and
                      toggle_y + button_height + button_spacing <= mouse_y <= 
                      toggle_y + 2 * button_height + button_spacing):
                    self.show_path = not self.show_path
                # Toggle labels
                elif (x_offset <= mouse_x <= x_offset + button_width and
                      toggle_y + 2 * (button_height + button_spacing) <= mouse_y <= 
                      toggle_y + 3 * button_height + 2 * button_spacing):
                    self.show_labels = not self.show_labels
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.paused_clicked:
                    self.paused_clicked = False
                    self.paused = not self.paused
                elif self.step_clicked:
                    self.step_clicked = False
                    self.paused = True
                    self.step = True
        pygame.event.pump()

    @staticmethod
    def close():
        """Closes the UI."""
        pygame.quit()