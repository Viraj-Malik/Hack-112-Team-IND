import FaceDetectionFinal

Final_Player=FaceDetectionFinal.Final_Player
game=int(input("Enter 1 for Snake Game and 2 for Flappy Bird: "))

if game==1:
    import pygame
    import random

    # Initialize pygame
    pygame.init()

    # Set up the display
    WIDTH, HEIGHT = 600, 400
    BLOCK_SIZE = 20
    GRID_WIDTH, GRID_HEIGHT = WIDTH // BLOCK_SIZE, HEIGHT // BLOCK_SIZE
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake Game")

    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)

    # Snake properties
    if Final_Player=="Viraj":
        SNAKE_SPEED = 10
    elif Final_Player=="Arnav":
        SNAKE_SPEED=2

    # Snake class
    class Snake:
        def __init__(self):
            self.length = 1
            self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
            self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
            self.color = GREEN

        def get_head_position(self):
            return self.positions[0]

        def move(self):
            cur = self.get_head_position()
            x, y = self.direction
            new = ((cur[0] + x) % GRID_WIDTH, (cur[1] + y) % GRID_HEIGHT)
            if len(self.positions) > 2 and new in self.positions[2:]:
                self.reset()
            else:
                self.positions.insert(0, new)
                if len(self.positions) > self.length:
                    self.positions.pop()

        def reset(self):
            self.length = 1
            self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
            self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

        def draw(self, surface):
            for p in self.positions:
                pygame.draw.rect(surface, self.color, (p[0] * BLOCK_SIZE, p[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        def handle_keys(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.direction != DOWN:
                        self.direction = UP
                    elif event.key == pygame.K_DOWN and self.direction != UP:
                        self.direction = DOWN
                    elif event.key == pygame.K_LEFT and self.direction != RIGHT:
                        self.direction = LEFT
                    elif event.key == pygame.K_RIGHT and self.direction != LEFT:
                        self.direction = RIGHT

    # Food class
    class Food:
        def __init__(self):
            self.position = (0, 0)
            self.color = RED
            self.randomize_position()

        def randomize_position(self):
            self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

        def draw(self, surface):
            pygame.draw.rect(surface, self.color, (self.position[0] * BLOCK_SIZE, self.position[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    # Directions
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def draw_grid(surface):
        for x in range(0, WIDTH, BLOCK_SIZE):
            pygame.draw.line(surface, WHITE, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, BLOCK_SIZE):
            pygame.draw.line(surface, WHITE, (0, y), (WIDTH, y))

    def main():
        clock = pygame.time.Clock()
        snake = Snake()
        food = Food()

        while True:
            WIN.fill(BLACK)
            draw_grid(WIN)
            snake.handle_keys()
            snake.move()
            snake.draw(WIN)
            food.draw(WIN)

            if snake.get_head_position() == food.position:
                snake.length += 1
                food.randomize_position()

            # Check for collision with walls
            if snake.get_head_position()[0] < 0 or snake.get_head_position()[0] >= GRID_WIDTH or \
                    snake.get_head_position()[1] < 0 or snake.get_head_position()[1] >= GRID_HEIGHT:
                print(score)
                snake.reset()

            pygame.display.update()
            clock.tick(SNAKE_SPEED)

    if __name__ == "__main__":
        main()

elif game==2:
    import pygame
    import random
    import cv2
    import numpy as np

    # Open Camera object
    cap = cv2.VideoCapture(0)

    # Decrease frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def nothing(x):
        pass

    # Creating a window for HSV track bars
    cv2.namedWindow('HSV_TrackBar')

    # Creating track bar
    cv2.createTrackbar('h', 'HSV_TrackBar', 0, 179, nothing)
    cv2.createTrackbar('s', 'HSV_TrackBar', 0, 255, nothing)
    cv2.createTrackbar('v', 'HSV_TrackBar', 0, 255, nothing)

    # Pygame initialization
    pygame.init()

    # Constants
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 600
    GRAVITY = 0.25 
    BIRD_JUMP = 4
    if Final_Player=="Arnav":
        PIPE_SPEED = 4
    else:
        PIPE_SPEED=1
    PIPE_GAP = 150
    PIPE_WIDTH = 50
    BIRD_WIDTH = 40
    BIRD_HEIGHT = 30
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # Constants for smoother motion
    MAX_VELOCITY = 5
    DAMPING = 0.95

    # Create screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")

    # Functions
    def draw_background():
        screen.fill(WHITE)  # Fill background with white color

    def draw_bird(x, y):
        pygame.draw.rect(screen, BLACK, (x, y, BIRD_WIDTH, BIRD_HEIGHT))  

    def draw_pipe(x, y, height):
        pygame.draw.rect(screen, BLACK, (x, 0, PIPE_WIDTH, y))
        pygame.draw.rect(screen, BLACK, (x, y + PIPE_GAP, PIPE_WIDTH, height - y - PIPE_GAP))

    def collision_detection(bird_x, bird_y, pipe_x, pipe_y):
        # Add a margin of error to the collision detection
        if bird_x + BIRD_WIDTH - 5 > pipe_x and bird_x + 5 < pipe_x + PIPE_WIDTH:
            if bird_y + 5 < pipe_y or bird_y + BIRD_HEIGHT - 5 > pipe_y + PIPE_GAP:
                return True
        print(score)
        return False

    # Game variables
    bird_x = 50
    bird_y = SCREEN_HEIGHT // 2
    bird_velocity = 0
    pipes = [{'x': SCREEN_WIDTH, 'y': random.randint(50, SCREEN_HEIGHT - 200)}]
    score = 0

    running = True

    # Main loop
    while running:
        # Limit bird velocity to prevent excessive speed
        if bird_velocity > MAX_VELOCITY:
            bird_velocity = MAX_VELOCITY

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Increase bird velocity instantly on key press
                    bird_velocity = -BIRD_JUMP

        # Move bird
        bird_y += bird_velocity
        bird_velocity += GRAVITY
        bird_velocity *= DAMPING  # Apply damping to gradually reduce velocity

        # Generate pipes
        if pipes[-1]['x'] < SCREEN_WIDTH - 200:
            pipes.append({'x': SCREEN_WIDTH, 'y': random.randint(50, SCREEN_HEIGHT - 200)})

        # Move pipes
        for pipe in pipes:
            pipe['x'] -= PIPE_SPEED

        # Remove off-screen pipes
        if pipes[0]['x'] < -PIPE_WIDTH:
            pipes.pop(0)
            score += 1

        # Capture frames from the camera
        ret, frame = cap.read()

        # Blur the image
        blur = cv2.blur(frame, (3, 3))

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

        # Kernel matrices for morphological transformation
        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Perform morphological transformations to filter out the background noise
        # Dilation increase skin color area
        # Erosion increase skin color area
        dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
        erosion = cv2.erode(dilation, kernel_square, iterations=1)
        dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
        filtered = cv2.medianBlur(dilation2, 5)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        median = cv2.medianBlur(dilation2, 5)
        ret, thresh = cv2.threshold(median, 127, 255, 0)

        # Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Check if contours are found
        if contours:
            # Find Max contour area (Assume that hand is in the frame)
            max_area = 100
            ci = 0
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    ci = i

            # Largest area contour
            cnts = contours[ci]

            # Find convex hull
            hull = cv2.convexHull(cnts)

            # Find convex defects
            hull2 = cv2.convexHull(cnts, returnPoints=False)
            defects = cv2.convexityDefects(cnts, hull2)

            # Get defect points and draw them in the original image
            FarDefect = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnts[s][0])
                end = tuple(cnts[e][0])
                far = tuple(cnts[f][0])
                FarDefect.append(far)
                cv2.line(frame, start, end, [0, 255, 0], 1)
                cv2.circle(frame, far, 10, [100, 255, 255], 3)

            # Find moments of the largest contour
            moments = cv2.moments(cnts)

            # Central mass of first order moments
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
            centerMass = (cx, cy)

            # Print center coordinates
            print("Center coordinates (x, y):", centerMass[0], centerMass[1])

            # Check for collision with pipes
            for pipe in pipes:
                if collision_detection(bird_x, bird_y, pipe['x'], pipe['y']):
                    running = False

            # Move bird
            bird_y = centerMass[1]

            # Draw center mass
            cv2.circle(frame, centerMass, 7, [100, 0, 255], 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'X: {centerMass[0]}, Y: {centerMass[1]}', (10, 30), font, 1, (255, 255, 255), 2)

            # Distance from each finger defect(finger webbing) to the center mass
            distanceBetweenDefectsToCenter = []
            for i in range(0, len(FarDefect)):
                x = np.array(FarDefect[i])
                centerMass = np.array(centerMass)
                distance = np.sqrt(np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
                distanceBetweenDefectsToCenter.append(distance)

            # Get an average of three shortest distances from finger webbing to center mass
            sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
            AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

            # Get fingertip points from contour hull
            # If points are in proximity of 80 pixels, consider as a single point in the group
            finger = []
            for i in range(0, len(hull) - 1):
                if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 80) or (
                        np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 80):
                    if hull[i][0][1] < 500:
                        finger.append(hull[i][0])

            # The fingertip points are 5 hull points with largest y coordinates
            finger = sorted(finger, key=lambda x: x[1])
            fingers = finger[0:5]

            # Calculate distance of each finger tip to the center mass
            fingerDistance = []
            for i in range(0, len(fingers)):
                distance = np.sqrt(
                    np.power(fingers[i][0] - centerMass[0], 2) + np.power(fingers[i][1] - centerMass[0], 2))
                fingerDistance.append(distance)

            # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
            # than the distance of average finger webbing to center mass by 130 pixels
            result = 0
            for i in range(0, len(fingers)):
                if fingerDistance[i] > AverageDefectDistance + 130:
                    result = result + 1

            # Print number of pointed fingers
            cv2.putText(frame, str(result), (100, 100), font, 2, (255, 255, 255), 2)

            # Print bounding rectangle
            x, y, w, h = cv2.boundingRect(cnts)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.drawContours(frame, [hull], -1, (255, 255, 255), 2)

        # Show final image
        cv2.imshow('Dilation', frame)

        # Update Pygame display
        draw_background()
        draw_bird(bird_x, bird_y)
        for pipe in pipes:
            draw_pipe(pipe['x'], pipe['y'], SCREEN_HEIGHT)
        pygame.display.flip()
        # Close the output video by pressing 'ESC'
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            print(score)
            break

    cap.release()
    cv2.destroyAllWindows()
