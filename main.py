import pygame
import random
import math
from settings import *
import tensorflow as tf
import numpy as np
import random
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Initialize Pygame
pygame.init()

clock = pygame.time.Clock()
delta_time = 0
time = 0

points = 0

grid_width = GRID_WIDTH
grid_height = GRID_HEIGHT
grid = np.zeros((grid_width, grid_height))

grid_max_values = np.zeros((grid_width, grid_height))

font = pygame.font.SysFont(None, 24)

player_grid_starting_position = [3, 3]
player_grid_position = player_grid_starting_position

# Setup the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Grid Game")

# Font for displaying text
pygame.font.init()
font = pygame.font.SysFont(None, 24)

# Player
player_position = np.array([player_grid_starting_position[0] * GRID_SIZE - GRID_SIZE / 2, player_grid_starting_position[1] * GRID_SIZE - GRID_SIZE / 2])
player_speed = np.array([0.0])
player_max_speed = 5
player_acceleration = 0.1
player_direction = np.array([0, 1]) # Initially facing upwards
player_rotation = 0
player_rotational_velocity = 0

num_columns = 0
columns = []

selected_action = 0

# Actions (assuming one-hot encoding)
actions = np.array([0, 1, 2, 3])
#actions[selected_action] = 1

flattened_grid = grid.flatten()


# Parameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
#epsilon_decay = 0.99
#episodes = 200
#run_episodes = 10

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
max_memory_length = 100000
batch_size = 32
update_after_actions = 4
update_target_network = 300

#input_vector = np.concatenate((flattened_grid, player_direction, player_speed, actions))

grid[4,1] = 1
grid[4,2] = 1
grid[4,3] = 1
grid[5,3] = 1
grid[6,3] = 1
grid[7,3] = 1
grid[8,3] = 1
grid[9,3] = 1

grid[4,9] = 1
grid[4,8] = 1
grid[4,7] = 1
grid[5,7] = 1
grid[6,7] = 1
grid[7,7] = 1
grid[8,7] = 1
grid[9,7] = 1

target_grid_position = (8, 5)

#grid[player_grid_starting_position[0], player_grid_starting_position[1]] = 2
grid[target_grid_position[0], target_grid_position[1]] = 3

# grid[5, 0] = 1
# grid[5, 1] = 1
# grid[5, 2] = 1
# grid[5, 3] = 1
# grid[5, 4] = 1
# grid[5, 5] = 1
# grid[5, 6] = 1
# grid[5, 7] = 1
# grid[5, 8] = 1
# grid[5, 9] = 1
# #grid[5, 10] = 1
# #grid[5, 11] = 1
#
# #grid[10, 9] = 1
# #grid[10, 10] = 1
# grid[10, 11] = 1
# grid[10, 12] = 1
# grid[10, 13] = 1
# grid[10, 14] = 1
# grid[10, 15] = 1
# grid[10, 16] = 1
# grid[10, 17] = 1
# grid[10, 18] = 1
# grid[10, 19] = 1
#
# grid[15, 0] = 1
# grid[15, 1] = 1
# grid[15, 2] = 1
# grid[15, 3] = 1
# grid[15, 4] = 1
# grid[15, 5] = 1
# grid[15, 6] = 1
# grid[15, 7] = 1
# grid[15, 8] = 1
# grid[15, 9] = 1
# #grid[15, 10] = 1
# #grid[15, 11] = 1

for x in range(0, 11):

    grid[x, 0] = 1
    columns.append([x, 0])

    grid[x, 10] = 1
    columns.append([x, 10])

    for y in range(0, 10):

        if 1 <= y <= 9 and (x == 0 or x == 10):

            grid[x, y] = 1
            columns.append([x, y])

        if grid[x, y] == 1:

                columns.append([x, y])

for _ in range(num_columns):
    while True:
        col_x = random.randint(0, grid_width - 1)
        col_y = random.randint(0, grid_height - 1)

        if grid[col_x, col_y] == 0 and (col_x, col_y) != player_grid_starting_position and (col_x, col_y) != target_grid_position:
            columns.append([col_x, col_y])
            grid[col_x, col_y] = 1  # Assuming columns are represented by the number '1'
            break


def draw_grid():
    for temp_x in range(GRID_WIDTH):
        for temp_y in range(GRID_HEIGHT):
            min_value = min(min(row) for row in grid_max_values)
            max_value = max(max(row) for row in grid_max_values) + 0.1
#            print(grid_max_values[temp_x, temp_y])

            pygame.draw.rect(screen, (0, 0, 255 * (grid_max_values[temp_x, temp_y] - min_value) / (max_value - min_value)), (temp_x * GRID_SIZE, temp_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
#            pygame.draw.rect(screen, (0, 0, 255), (temp_x * GRID_SIZE, temp_y * GRID_SIZE + 2 * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            rect = pygame.Rect(temp_x * GRID_SIZE, temp_y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)



def draw_player():
    pygame.draw.rect(screen, GREEN, (int(player_position[0]) - PLAYER_SIZE / 2, int(player_position[1] - PLAYER_SIZE / 2), GRID_SIZE, GRID_SIZE))
    pygame.draw.circle(screen, RED, (int(player_position[0]), int(player_position[1])), 2)
    end_pos = [player_position[0] + player_direction[0] * PLAYER_RADIUS, player_position[1] + player_direction[1] * PLAYER_RADIUS]
    pygame.draw.line(screen, BLACK, (int(player_position[0]), int(player_position[1])), end_pos, 4)


def draw_target():
    pygame.draw.rect(screen, RED, (target_grid_position[0] * GRID_SIZE, target_grid_position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))


def draw_pillars():

    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            # Check if the cell has a column
            if grid[x, y] == 1:
                # Draw a rectangle for the column
                rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, (255, 255, 255), rect)


def move():

    global player_grid_position, player_position, player_rotation, player_speed, player_direction, player_rotational_velocity

    player_rotation = math.degrees(math.atan2(player_direction[1], player_direction[0]))

    player_rotational_velocity *= ROTATION_DAMPING
    player_rotation -= player_rotational_velocity

    rad_rotation = math.radians(player_rotation)
    player_direction = np.array([math.cos(rad_rotation), math.sin(rad_rotation)])

    player_speed *= DAMPING

    next_pos = np.add(player_direction * player_speed, player_position)

    # Collision with right or left border
    if next_pos[0] + PLAYER_SIZE > SCREEN_WIDTH:
        next_pos[0] = SCREEN_WIDTH - PLAYER_SIZE
        player_direction[0] = -abs(player_direction[0])
        player_speed *= 0.25

    elif next_pos[0] - PLAYER_SIZE < 0:
        next_pos[0] = PLAYER_SIZE
        player_direction[0] = abs(player_direction[0])
        player_speed *= 0.25

    # Collision with bottom or top border
    if next_pos[1] + PLAYER_SIZE > SCREEN_HEIGHT:
        next_pos[1] = SCREEN_HEIGHT - PLAYER_SIZE
        player_direction[1] *= -1
        player_speed *= 0.25
    elif next_pos[1] - PLAYER_SIZE / 2 < 0:
        next_pos[1] = PLAYER_SIZE
        player_direction[1] *= -1
        player_speed *= 0.25

    #player_position = np.add(player_direction * player_speed, player_position)

    # Collision with columns
    # for column in columns:
    #     column_rect = pygame.Rect(column[0] * GRID_SIZE, column[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    #     player_rect = pygame.Rect(int(next_pos[0]) - GRID_SIZE / 2, int(next_pos[1]) - GRID_SIZE / 2,
    #                               GRID_SIZE, GRID_SIZE)
    #
    #     if player_rect.colliderect(column_rect):
    #         horizontal_collision = vertical_collision = False
    #
    #         # Check primary direction of movement
    #         if abs(player_direction[0]) > abs(player_direction[1]):  # Primary movement is horizontal
    #             if player_direction[0] > 0 and player_rect.right > column_rect.left:
    #                 next_pos[0] = column_rect.left - GRID_SIZE / 2
    #                 player_direction[0] *= -1
    #                 horizontal_collision = True
    #             elif player_direction[0] < 0 and player_rect.left < column_rect.right:
    #                 next_pos[0] = column_rect.right + GRID_SIZE / 2
    #                 player_direction[0] *= -1
    #                 horizontal_collision = True
    #
    #         if abs(player_direction[1]) >= abs(player_direction[
    #                                                0]) or not horizontal_collision:  # Primary movement is vertical or no horizontal collision
    #             if player_direction[1] > 0 and player_rect.bottom > column_rect.top:
    #                 next_pos[1] = column_rect.top - GRID_SIZE / 2
    #                 player_direction[1] *= -1
    #                 vertical_collision = True
    #             elif player_direction[1] < 0 and player_rect.top < column_rect.bottom:
    #                 next_pos[1] = column_rect.bottom + GRID_SIZE / 2
    #                 player_direction[1] *= -1
    #                 vertical_collision = True
    #
    #         if horizontal_collision and vertical_collision:
    #             # Handle special case of diagonal collision
    #             # You may choose to resolve it in a specific way, e.g., reduce speed, stop, etc.
    #             pass
    #
    #         break  # Exit loop after handling collision

    for column in columns:
        column_rect = pygame.Rect(column[0] * GRID_SIZE, column[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        player_rect = pygame.Rect(int(next_pos[0]) - PLAYER_SIZE / 2, int(next_pos[1]) - PLAYER_SIZE / 2,
                                  GRID_SIZE, GRID_SIZE)

        if player_rect.colliderect(column_rect):

            player_speed *= 0.25
            # Calculate overlap distances on each side
            overlap_left = player_rect.right - column_rect.left
            overlap_right = column_rect.right - player_rect.left
            overlap_top = player_rect.bottom - column_rect.top
            overlap_bottom = column_rect.bottom - player_rect.top

            # Determine the minimum overlap to find the collision edge
            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

            if min_overlap == overlap_left:
                next_pos[0] -= overlap_left  # Move player to the left
                player_direction[0] *= -1
            elif min_overlap == overlap_right:
                next_pos[0] += overlap_right  # Move player to the right
                player_direction[0] *= -1
            elif min_overlap == overlap_top:
                next_pos[1] -= overlap_top  # Move player upwards
                player_direction[1] *= -1
            elif min_overlap == overlap_bottom:
                next_pos[1] += overlap_bottom  # Move player downwards
                player_direction[1] *= -1

            break  # Exit loop after handling collision

    player_position = next_pos
    player_grid_position[0] = int(player_position[0] // GRID_SIZE)
    player_grid_position[1] = int(player_position[1] // GRID_SIZE)

#    if not result:
#        player_position += next_step

#    if not check_player_collision(player_position, dy=next_step[1]):
#        player_position += next_step

def game_step(selected_action):

    global player_position, actions, player_rotation, player_speed, player_direction, player_rotational_velocity

#    action = random.randint(0, 4);

    # Actions (assuming one-hot encoding)
    actions = np.zeros(4)
    actions[selected_action] = 1

#    print(selected_action)

    if selected_action == 1:
        player_rotational_velocity += 1
    if selected_action == 2:
        player_rotational_velocity -= 1
    if selected_action == 3:
        player_speed += 1

    move()


def keyboard_controls():

    global player_position, selected_action, player_rotation, player_speed, player_direction, player_rotational_velocity

#    print(player_direction)

    key_state = pygame.key.get_pressed()
#    vel = player_speed * delta_time
#    rot_vel = player_rotation * delta_time
    #next_step = player_position
    #print(next_step)

    selected_action = 3

    #
    if key_state[KEYS['TURN_LEFT']]:
        selected_action = 1
    if key_state[KEYS['TURN_RIGHT']]:
        selected_action = 2
    if key_state[KEYS['FORWARD']]:
        selected_action = 3

    return selected_action

#    move()


def check_reached_target():

    global points, player_position, player_speed

    reward = 0

    if (target_grid_position[0] * GRID_SIZE <= player_position[0] <= target_grid_position[0] * GRID_SIZE + GRID_SIZE) and (target_grid_position[1] * GRID_SIZE <= player_position[1] <= target_grid_position[1] * GRID_SIZE + GRID_SIZE):

        points += 1
        reward = 100
        player_position = [3 * GRID_SIZE - GRID_SIZE / 2, 3 * GRID_SIZE - GRID_SIZE / 2]
        player_speed = np.array([0.0])

    return reward

def draw_points():

    global points

    text_surface = font.render(f'Points: {points}', True, (0, 0, 0))  # White color

    # Position the text in the upper right corner
    text_rect = text_surface.get_rect()
#    text_rect.topright = (SCREEN_WIDTH - 10, 10)  # Adjust padding as needed

    # Draw everything
#    screen.fill((0, 0, 0))  # Clear screen
    screen.blit(text_surface, text_rect)  # Draw the text

def get_current_state():

    global input_vector, model, grid, player_direction, player_grid_position, player_speed, actions

    flattened_grid = grid.flatten()
#    print(grid)#    print(flattened_grid)
#    print(player_direction)
#    print(player_speed)
#    print(actions)

    input_vector = np.concatenate((flattened_grid, player_grid_position, player_direction, player_speed))

    return input_vector

def get_temp_state(temp_player_x, temp_player_y, temp_player_rad, temp_player_speed):

    global grid, player_grid_position

    flattened_grid = grid.flatten()
    temp_player_direction = np.array([math.cos(temp_player_rad), math.sin(temp_player_rad)])
    temp_player_position = ([temp_player_x, temp_player_y])
    temp_player_speed = ([temp_player_speed])
#    print(grid)
#    print(flattened_grid)
#    print(player_direction)
#    print(player_speed)
#    print(actions)

    temp_input_vector = np.concatenate((flattened_grid, temp_player_direction, temp_player_position, temp_player_speed))

    return temp_input_vector


def build_model():

    global model, actions, input_vector

    # Neural network for the agent
    model = Sequential([
        Dense(32, input_dim=input_vector.size, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(actions.size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))


def get_q_values(state):

    global model

    state = np.reshape((state, [1, state.size]))
    q_values = model.predict(state)
    return q_values[0]


def train_model(training_episodes, epsilon_decay, initial_epsilon, agent_control):

    global epsilon, actions, pygame, player_position, player_grid_position, player_speed, player_direction

    epsilon = initial_epsilon

    for episode in range(0, training_episodes):

        episode_reward = 0

        for timestep in range(1, update_target_network + 1):

            episode_reward -= 1

#            selected_action = 0

#            print(agent_control)

            if agent_control:

                if random.random() <= epsilon:
                    actions = np.array([0, 1, 2, 3])
#                    print(actions.size)

    #                print(f"actions are {actions}")
                    selected_action = np.random.choice(actions)
    #                print(f"selected action is {selected_action}")
                else:
                    action_probs = model.predict(input_vector.reshape(1, -1), verbose=0)
                    selected_action = np.argmax(action_probs[0])
#                    print(selected_action)

            else:

                pass
#                selected_action = human_action

#            print(selected_action)

#            selected_action = keyboard_controls()

            # Apply the sampled action in our environment
            game_step(selected_action)
            reached_target = 0

            reached_target = check_reached_target()
            episode_reward += reached_target
            #    player_pos[0] += 1
            #    move_player()
            screen.fill((0, 0, 0))
            draw_grid()
            draw_player()
            draw_target()
            draw_pillars()
            draw_points()
            pygame.display.flip()
#            pygame.time.Clock().tick()

            state_next = get_current_state()
            if np.any(np.isnan(state_next)) or not np.all(np.isfinite(state_next)):
                print("Input data contains NaN or infinite values")

            state_next = np.array(state_next)

#            episode_reward = 0

#            print(f"episode {episode}; timestep {timestep}; episode reward {episode_reward}")
            # Save actions and states in replay buffer
            action_history.append(selected_action)
            state_history.append(input_vector)
            state_next_history.append(state_next)
            done_history.append(reached_target)
            rewards_history.append(episode_reward)
            state = state_next

            if True:
                # Update every fourth frame and once batch size is over 32
                if len(action_history) > batch_size and timestep % update_after_actions == 0:
                    indices = np.random.choice(range(len(done_history)), size=batch_size)

                    #                for i in indices:
                    #                    print(len(state_history[i]))

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                    # Build updated Q-values for the sampled future states
                    future_rewards = model.predict(state_next_sample)
#                    print(future_rewards)
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, actions.size)

                    with tf.GradientTape() as tape:
                        q_values = model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        loss = tf.keras.losses.mean_squared_error(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if timestep % update_target_network == 0:
                # update the target network parameters
                model.set_weights(model.get_weights())

            if reached_target:
                break

        rand_xx = random.randint(1,9)
        rand_yy = random.randint(1, 9)

        for rand_x in range(rand_xx - 1, rand_xx + 1):
            for rand_y in range(rand_yy - 1, rand_yy + 1):

        # rand_x = random.randint(0, 9)
        # rand_y = random.randint(0, 9)

                temp_q_values = np.zeros(4)

                for rad in range(0, 4):

#                rand_rad = random.uniform(0, 2 * np.pi)
#                rand_speed = random.uniform(0, 5)
                    rand_rad = rad * np.pi / 2
                    rand_speed = 1

                    temp_input_vector = get_temp_state(rand_x, rand_y, rand_rad, rand_speed)

                    temp_q_values[rad] = np.max(
                        model.predict(temp_input_vector.reshape(1, -1), verbose=1))

#                    grid_max_values[rand_x, rand_y] = np.max(
#                        model.predict(temp_input_vector.reshape(1, -1), verbose=1))

                grid_max_values[rand_x, rand_y] = np.mean(temp_q_values)

                if rand_x == 9 and rand_y == 9:

                    print(grid_max_values[rand_x, rand_y])


        player_position = [3 * GRID_SIZE - GRID_SIZE / 2, 3 * GRID_SIZE - GRID_SIZE / 2]
        player_grid_position = [3, 3]
        player_speed = np.array([0.0])
        player_direction = np.array([0,1])

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f'Training Episode: {episode}, Epsilon: {epsilon}, Total Reward: {episode_reward}')

    pygame.quit()


get_current_state()
build_model()

# Nested loops
# for x in range(GRID_WIDTH):
#     for y in range(GRID_HEIGHT):
#         grid_max_values[x, y] = random.randrange(-10000, 10000)

#model = models.load_model("game.keras")

train_model(6000, .9997, initial_epsilon=1, agent_control=True)

#train_model(200, .997, initial_epsilon=1, agent_control=True)

model.save("game2.keras")

if False:
    # Main game loop
    running = True

    while running:

        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                running = False

        keyboard_controls()
    #    agent_controls()

        check_reached_target()

        get_current_state()

        #    player_pos[0] += 1
    #    move_player()
#        screen.fill((0, 0, 0))
#        draw_grid()
#        draw_player()
#        draw_target()
#        draw_pillars()
#        draw_points()
#        pygame.display.flip()
#        pygame.time.Clock().tick(10)

    pygame.quit()
