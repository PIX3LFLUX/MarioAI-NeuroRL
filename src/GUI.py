import pygame
import sys
import subprocess

# Initialize Pygame
pygame.init()

# Set up the window
width = pygame.display.Info().current_w
height = pygame.display.Info().current_h
monitor_size = (width, height)
screen = pygame.display.set_mode(monitor_size, pygame.FULLSCREEN)
pygame.display.set_caption("Game Mode Selection")


# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up fonts
font = pygame.font.Font(None, 30)

def level_selector():
    running = True
    selected_level = None

    while running:
        screen.fill(BLACK)

        # Display text
        level_text = font.render("Select a level:", True, WHITE)
        level_1_text = font.render("1. Level 1-1", True, WHITE)
        level_2_text = font.render("2. Level 1-2", True, WHITE)
        level_3_text = font.render("3. Level 1-3", True, WHITE)

        # Position the text
        level_rect = level_text.get_rect(center=(width // 2, height // 4))
        level_1_rect = level_1_text.get_rect(center=(width // 2, height // 2))
        level_2_rect = level_2_text.get_rect(center=(width // 2, height // 2 + 50))
        level_3_rect = level_3_text.get_rect(center=(width // 2, height // 2 + 100))

        # Blit the text onto the screen
        screen.blit(level_text, level_rect)
        screen.blit(level_1_text, level_1_rect)
        screen.blit(level_2_text, level_2_rect)
        screen.blit(level_3_text, level_3_rect)

        # Update the display
        pygame.display.flip()

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selected_level = "SuperMarioBros-1-1-v0"
                    running = False
                elif event.key == pygame.K_2:
                    selected_level = "SuperMarioBros-1-2-v0"
                    running = False
                elif event.key == pygame.K_3:
                    selected_level = "SuperMarioBros-1-3-v0"
                    running = False
                    
        #for event in pygame.event.get():
        #    if event.type == pygame.KEYDOWN:
        #        if event.key == K_ESCAPE:
        #            pygame.quit()

    return selected_level

# Game loop
running = True
selected_mode = None
selected_training_mode = None
selected_level = None
selected_test_model = None

while running:
    screen.fill(BLACK)

    # Display text
    title_text = font.render("Select Game Mode", True, WHITE)
    ai_text = font.render("1. AI Player", True,WHITE)
    self_text = font.render("2. Human Player", True, WHITE)

    # Position the text
    title_rect = title_text.get_rect(center=(width // 2, height // 4))
    ai_rect = ai_text.get_rect(center=(width // 2, height // 2))
    self_rect = self_text.get_rect(center=(width // 2, height // 2 + 50))

    # Blit the text onto the screen
    screen.blit(title_text, title_rect)
    screen.blit(ai_text, ai_rect)
    screen.blit(self_text, self_rect)

    # Update the display
    pygame.display.flip()

    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                print("AI-Player mode selected.")
                selected_mode = "AI-Player"
                running = False
            elif event.key == pygame.K_2:
                print("Human-Player mode selected.")
                selected_mode = "Human-Player"
                running = False
                
    #for event in pygame.event.get():
    #    if event.type == pygame.KEYDOWN:
    #        if event.key == K_ESCAPE:
    #            pygame.quit()

# Check if a mode was selected
if selected_mode == "AI-Player":
    running = True
    while running:
        screen.fill(BLACK)
    
        # Display text 
        title_text = font.render("Select pre-trained Model", True, WHITE)
        model1_text = font.render("1. 1000 iterations (Level 1-1)", True, WHITE)
        model2_text = font.render("2. 2000 iterations (Level 1-1)", True, WHITE)
        model3_text = font.render("3. 8000 iterations (Level 1-1)", True, WHITE)
        model4_text = font.render("4. Expert model (Level 1-1)", True, WHITE)
        
        # Position the text 
        title_rect = title_text.get_rect(center=(width // 2, height // 4))
        model1_rect = model1_text.get_rect(center=(width // 2, height // 2))
        model2_rect = model2_text.get_rect(center=(width // 2, height // 2 + 50))
        model3_rect = model3_text.get_rect(center=(width // 2, height // 2 + 100))
        model4_rect = model4_text.get_rect(center=(width // 2, height // 2 + 150))
        
        # Blit the text onto the screen
        screen.blit(title_text, title_rect)
        screen.blit(model1_text, model1_rect)
        screen.blit(model2_text, model2_rect)
        screen.blit(model3_text, model3_rect)
        screen.blit(model4_text, model4_rect)
        
        # Update the display
        pygame.display.flip()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    print("1000 iterations model selected.")
                    selected_test_model = "model1"
                    running = False
                elif event.key == pygame.K_2:
                    print("2000 iterations model selected.")
                    selected_test_model = "model2"
                    running = False
                elif event.key == pygame.K_3:
                    print("8000 iterations model selected.")
                    selected_test_model = "model3"
                    running = False
                elif event.key == pygame.K_4:
                    print("Expert model selected.")
                    selected_test_model = "model4"
                    running = False
                    
        #for event in pygame.event.get():
        #    if event.type == pygame.KEYDOWN:
        #        if event.key == K_ESCAPE:
        #            pygame.quit()
    
# Check if a training mode was selected
if selected_test_model or (selected_mode == "Human-Player"):
    selected_level = level_selector()

automatic_mode = "False"

# Check if a training mode and level were selected
if (selected_test_model or (selected_mode == "Human-Player")) and selected_level:
    print(f"{selected_mode} mode selected for level {selected_level}.")
    pygame.quit()
    if selected_mode == "AI-Player":
        subprocess.call(["python", "test_models.py", selected_level, selected_test_model, automatic_mode])
    elif selected_mode == "Human-Player":
        subprocess.call(["python", "human_player.py", selected_level])
