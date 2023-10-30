import pygame
import sys
import subprocess

# Initialize Pygame
pygame.init()

# Set up the window
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Game Mode Selection")

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up fonts
font = pygame.font.Font(None, 30)

def level_selector():
    running = True
    selected_level = None

    while running:
        screen.fill(WHITE)

        # Display text3
        level_text = font.render("Select a level:", True, BLACK)
        level_1_text = font.render("1. Level 1-1", True, BLACK)
        level_2_text = font.render("2. Level 1-2", True, BLACK)
        level_3_text = font.render("3. Level 1-3", True, BLACK)

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

    return selected_level

# Initialize Pygame
pygame.init()

# Set up the window
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Game Mode Selection")

# Set up colors
WHITE = (0, 0, 0)
BLACK = (255, 255, 255)

# Set up fonts
font = pygame.font.Font(None, 30)

# Game loop
running = True
selected_mode = None
selected_training_mode = None
selected_level = None

while running:
    screen.fill(WHITE)

    # Display text
    title_text = font.render("Select Game Mode", True, BLACK)
    ai_text = font.render("1. AI Player", True, BLACK)
    self_text = font.render("2. Human Player", True, BLACK)

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
                print("AI Player mode selected.")
                selected_mode = "AI Player"
                running = False
            elif event.key == pygame.K_2:
                print("Self-Player mode selected.")
                selected_mode = "Self-Player"
                running = False

# Check if a mode was selected
if selected_mode:
    running = True
    while running:
        screen.fill(WHITE)

        # Display text
        title_text = font.render("Select Training Mode", True, BLACK)
        train_text = font.render("1. Train Models", True, BLACK)
        test_text = font.render("2. Test Models", True, BLACK)

        # Position the text
        title_rect = title_text.get_rect(center=(width // 2, height // 4))
        train_rect = train_text.get_rect(center=(width // 2, height // 2))
        test_rect = test_text.get_rect(center=(width // 2, height // 2 + 50))

        # Blit the text onto the screen
        screen.blit(title_text, title_rect)
        screen.blit(train_text, train_rect)
        screen.blit(test_text, test_rect)

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
                    print("Train Models mode selected.")
                    selected_training_mode = "Train Models"
                    running = False
                elif event.key == pygame.K_2:
                    print("Test Models mode selected.")
                    selected_training_mode = "Test Models"
                    running = False

# Check if a training mode was selected
if selected_training_mode:
    selected_level = level_selector()

# Check if a training mode and level were selected
if selected_training_mode and selected_level:
    print(f"{selected_mode} mode selected for level {selected_level} with {selected_training_mode} mode.")
    pygame.quit()
    if selected_mode == "AI Player":
        if selected_training_mode == "Train Models":
            subprocess.call(["python", "Train_Models.py", selected_level])
        elif selected_training_mode == "Test Models":
            subprocess.call(["python", "Test_Models.py", selected_level])
    elif selected_mode == "Self-Player":
        subprocess.call(["python", "Human_Player.py", selected_level])
