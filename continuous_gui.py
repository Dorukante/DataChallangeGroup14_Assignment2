from time import perf_counter

import numpy as np
import pymunk
from pygame import gfxdraw

import pygame
import sys


class ContinuousGUI:
    # Colors
    COLOR_BG = (30, 30, 30)
    COLOR_AGENT = (0, 255, 0)
    COLOR_OBSTACLE = (200, 0, 0)
    COLOR_GOAL = (0, 0, 255)

    INFO_NAME_MAP = [
        ("cumulative_reward", "Cumulative reward:"),
        ("total_steps", "Total steps:"),
        ("total_failed_move", "Total failed moves:"),
        # ("total_targets_reached", "Total targets reached:"),
        ("fps", "FPS:"),
    ]

    def __init__(self, extents: tuple[int, int],
            window_size: tuple[int, int] = (1152, 768)):
        """Provides a GUI to show what is happening in the environment.

        Args:
            grid_size: (n_cols, n_rows) in the grid.
            window_size: The size of the pygame window. (width, height).
        """
        self.grid_size = extents

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Data Intelligence Challenge 2024")
        self.clock = pygame.time.Clock()

        self.stats = self._reset_stats()

        self.grid_panel_size = (int(window_size[0] * 0.75), window_size[1])
        self.info_panel_rect = pygame.Rect(
            self.grid_panel_size[0],
            0,
            window_size[0] - self.grid_panel_size[0],
            window_size[1]
        )
        self.last_agent_pos = None

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

        self._initial_render()

    def reset(self):
        """Called during the reset method of the environment."""
        self.stats = self._reset_stats()
        self._initial_render()

    @staticmethod
    def _reset_stats():
        return {"total_targets_reached": 0,
                "total_failed_move": 0,
                "fps": "0.0",
                "total_steps": 0,
                "cumulative_reward": 0}

    def _initial_render(self):
        """Initial render of the environment. Also shows loading text."""
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Display the loading text
        font = pygame.font.Font(None, 36)
        text = font.render("Loading environment...", True, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery

        # Blit the text onto the background surface
        background.blit(text, textpos)
        # Blit the background onto the window
        update_rect = self.window.blit(background, background.get_rect())
        # Tell pygame to update the display where the window was blit-ed
        pygame.display.update(update_rect)

    @staticmethod
    def _downsample_rect(rect: pygame.Rect, scalar: float) -> pygame.Rect:
        """Downsamples the given rectangle by a scalar."""
        x = rect.x * scalar
        y = rect.y * scalar
        width = rect.width * scalar
        height = rect.height * scalar
        return pygame.Rect(x, y, width, height)

    def _draw_button(self, surface: pygame.Surface, text: str,
            rect: pygame.Rect, color: tuple[int, int, int],
            text_color: tuple[int, int, int] = (0, 0, 0)):
        """Draws a button on the given surface."""
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, width=1)

        font = pygame.font.Font(None, int(self.scalar / 2))
        text = font.render(text, True, text_color)
        textpos = text.get_rect()
        textpos.centerx = rect.centerx
        textpos.centery = rect.centery
        surface.blit(text, textpos)

    def _draw_info(self, surface) -> tuple[pygame.Rect, pygame.Rect]:
        """Draws the info panel on the surface.

        Returns:
            The rect of the pause button and the step button.
        """
        x_offset = self.grid_panel_size[0] + 20
        y_offset = 50

        col_width = 200
        row_height = 30

        font = pygame.font.Font(None, 24)
        for row, (key, name) in enumerate(self.INFO_NAME_MAP):
            y_pos = y_offset + (row * row_height)
            text = font.render(name, True, (0, 0, 0))
            textpos = text.get_rect()
            textpos.x = x_offset
            textpos.y = y_pos
            surface.blit(text, textpos)

            text = font.render(f"{self.stats[key]}", True, (0, 0, 0))
            textpos = text.get_rect()
            textpos.x = x_offset + col_width
            textpos.y = y_pos
            surface.blit(text, textpos)

        button_row = len(self.INFO_NAME_MAP) + 1
        # Draw a button to pause the simulation
        pause_rect = pygame.Rect(x_offset,
            y_offset + (button_row * row_height) + 50,
            200,
            50)

        clicked_color = (155, 155, 155)
        color = (255, 255, 255)

        self._draw_button(surface, "Resume" if self.paused else "Pause",
            pause_rect,
            clicked_color if self.paused_clicked else color,
            (0, 0, 0))

        # Draw a button to step through the simulation
        step_rect = pygame.Rect(x_offset,
            y_offset + (button_row * row_height) + 110,
            200,
            50)
        self._draw_button(surface, "Step", step_rect,
            clicked_color if self.step_clicked else color,
            (0, 0, 0))

        return pause_rect, step_rect

    def render(self, env, reward: float = 0, is_single_step: bool = False):
        """Render the environment.
        """
        info = env.info

        self.frame_count += 1
        self.frame_count %= 10

        curr_time = perf_counter()
        fps = 1 / (curr_time - self.last_render_time)
        self.last_render_time = curr_time

        self.last_10_fps[self.frame_count] = fps
        self.stats["fps"] = f"{sum(self.last_10_fps) / 10:.1f}"
        self.stats["total_targets_reached"] += int(info["target_reached"])

        if (not self.paused and not self.step) or is_single_step:
            self.stats["total_steps"] += 1

        failed_move = 1 - int(info["agent_moved"])
        self.stats["total_failed_move"] += failed_move

        self.stats["cumulative_reward"] += reward

        # Create a surface to actually draw on
        background = pygame.Surface(self.window.get_size()).convert()
        background.fill((250, 250, 250))

        # Calculate grid offset
        grid_width = self.scalar * env.extents[0]
        grid_height = self.scalar * env.extents[1]
        x_offset = (self.grid_panel_size[0] / 2) - (grid_width / 2)
        y_offset = (self.grid_panel_size[1] / 2) - (grid_height / 2)

        # Draw the background for the info panel
        background.fill((238, 241, 240), self.info_panel_rect)


        # draw agent circle
        agent_pos = env.agent_body.position
        gfxdraw.filled_circle(background, int(agent_pos.x), int(agent_pos.y), int(env.AGENT_RADIUS), self.COLOR_AGENT)

        # draw goals
        for goal_obj in env.current_goals.values():
            goal = goal_obj.body.position  # use body position in case they are moving
            gfxdraw.filled_circle(background, int(goal.x), int(goal.y), int(env.GOAL_RADIUS), self.COLOR_GOAL)

        # draw obstacles
        for obstacle_body, obstacle_shape in env.obstacles:
            if isinstance(obstacle_shape, pymunk.Poly):
                # Draw polygon obstacles
                B = obstacle_body.position
                points = [(int(v.x + B.x), int(v.y + B.y)) for v in obstacle_shape.get_vertices()]
                pygame.draw.polygon(background, self.COLOR_OBSTACLE, points)

        sensor_distance = env.agent_state.front_sensor_distance
        sensor_type = env.agent_state.front_sensor_type

        sensor_end = (
            int(agent_pos.x + np.cos(env.agent_body.angle) * sensor_distance),
            int(agent_pos.y + np.sin(env.agent_body.angle) * sensor_distance)
        )
        pygame.draw.line(background, (255, 0, 0), (int(agent_pos.x), int(agent_pos.y)), sensor_end, 2)
        font = pygame.font.SysFont("Arial", 14)
        sensor_text = f"Sensor: {sensor_distance:.1f} px | Type: {sensor_type}"
        text_surface = font.render(sensor_text, True, (0, 0, 0))
        background.blit(text_surface, (10, self.window.get_height() - 20))

        pause_rect, step_rect = self._draw_info(background)

        # Blit the surface onto the window
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

        # Parse events that happened since the last render step
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Detect click events
                if pause_rect.collidepoint(event.pos):
                    self.paused_clicked = True
                elif step_rect.collidepoint(event.pos):
                    self.step_clicked = True
            elif event.type == pygame.MOUSEBUTTONUP:
                # Only do the action on mouse button up, as is expected.
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