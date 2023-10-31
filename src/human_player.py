import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import sys
import subprocess

# Transformationen des Eingangsenvironments
def make_env(level):
    env = gym_super_mario_bros.make(level)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env

def run_Self_Player():
    
    level = sys.argv[1]
    
    pygame.init()
    
    # Font settings for displaying text
    font = pygame.font.Font(None, 44)
    text_color = (255, 255, 255)

    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Create a dictionary to map controller buttons to actions
    buttons_to_action = {
        1: 0b00000010,  # Button 0
        2: 0b00000001  # Button 1
    }

    # Create a dictionary to map D-pad directions to actions
    d_pad_to_action = {
        (0, 1): 0b00010000,  # D-pad Up
        (0, -1): 0b00100000,  # D-pad Down
        (1, 0): 0b10000000,  # D-pad Right
        (-1, 0): 0b01000000 # D-pad Left
    }

    env = gym_super_mario_bros.make(level)

    # Initialize Pygame
    pygame.init()

    # Set the dimensions of the pygame window
    window_width = pygame.display.Info().current_w
    window_height = pygame.display.Info().current_h
    window = pygame.display.set_mode((window_width, window_height), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    
    small_window_width = 1600
    small_window_height = 1200

    env.reset()
    
    total_reward = 0

    steps = 0
    action = 0
    
    # Clear the screen with any color
    window.fill((0, 0, 0))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button in buttons_to_action:
                    action |= buttons_to_action[event.button]
                    print("Action:", action)
            elif event.type == pygame.JOYBUTTONUP:
                # A button has been released
                if event.button in buttons_to_action:
                    action &= ~buttons_to_action[event.button]
                else:
                    action &= ~sum(buttons_to_action.values())

            elif event.type == pygame.JOYAXISMOTION:
            	axis_value = event.value
            	axis_id = event.axis
            	action = d_pad_to_action[(axis_value, axis_id)]

            elif event.type == pygame.QUIT:
                # The window has been closed
                break
                
        steps += 1

        state_next, reward, terminal, info = env.step(action)
        total_reward += reward
        
        screen = env.render(mode='rgb_array')
        screen = pygame.surfarray.make_surface(screen)
        screen = pygame.transform.flip(screen, True, False)  # Flip vertically
        screen = pygame.transform.rotate(screen, 90)  # Rotate counterclockwise by 90 degrees
        screen = pygame.transform.scale(screen, (small_window_width, small_window_height))
        
        window.blit(screen, (window_width // 2 - small_window_width // 2, window_height // 2 - small_window_height // 2)) # Adjust the position as needed
        
        # Show Level and Model
        text2_surface = font.render("Level: {}".format(level), True, text_color)
        window.blit(text2_surface, (window_width  // 2 + small_window_width // 2 - 450, window_height // 2 - small_window_height // 2 + 10))
        
        pygame.display.flip()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
        
        if terminal:
            break
        
    print("Total Reward is {}".format(total_reward))
    
    score_font = pygame.font.Font(None, 36) 

    # Display the score in the score window
    score_text = score_font.render("Total Reward: {}".format(total_reward), True, (255, 255, 255))
    window.fill((0, 0, 0))
    window.blit(score_text, (window_width // 2 - score_text.get_width() // 2,
                               window_height // 2 - score_text.get_height() // 2))
    pygame.display.flip()

    # Wait for a button press to close the score window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pygame.quit()
                subprocess.call(["python", "GUI.py"])
                sys.exit()
            
run_Self_Player()
