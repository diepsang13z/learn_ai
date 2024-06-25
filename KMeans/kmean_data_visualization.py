import pygame
import math

from sklearn.cluster import KMeans
from random import randint


def distance(
    p1: list[int, int],
    p2: list[int, int],
) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def main():
    pygame.init()

    screen = pygame.display.set_mode((1200, 700))
    pygame.display.set_caption('kmeans visualization')

    running = True

    clock = pygame.time.Clock()

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (147, 153, 35)
    PURPLE = (255, 0, 255)
    SKY = (0, 255, 255)
    ORANGE = (255, 125, 25)
    GRAPE = (100, 25, 125)
    GRASS = (55, 155, 65)

    CLUSTER_COLORS = [
        RED,
        GREEN,
        BLUE,
        YELLOW,
        PURPLE,
        SKY,
        ORANGE,
        GRAPE,
        GRASS,
    ]

    BACKGROUND = (214, 214, 214)
    BACKGROUND_PANEL = (249, 255, 230)

    font_md = pygame.font.SysFont('sans', 40)
    font_sm = pygame.font.SysFont('sans', 20)

    text_plus = font_md.render('+', True, WHITE)
    text_mius = font_md.render('-', True, WHITE)
    text_run = font_md.render('Run', True, WHITE)
    text_random = font_md.render('Random', True, WHITE)
    text_algorithm = font_md.render('Algorithm', True, WHITE)
    text_reset = font_md.render('Reset', True, WHITE)

    k = 0
    sse = 0
    points = []
    clusters = []
    labels = []

    while running:
        clock.tick(60)
        screen.fill(BACKGROUND)

        # Draw interface

        # Draw panel
        pygame.draw.rect(screen, BLACK, (50, 50, 700, 500))
        pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 690, 490))

        # K button +
        pygame.draw.rect(screen, BLACK, (850, 50, 50, 50))
        screen.blit(text_plus, (865, 50))

        # K button -
        pygame.draw.rect(screen, BLACK, (950, 50, 50, 50))
        screen.blit(text_mius, (970, 50))

        # K value
        text_k = font_md.render('K = ' + str(k), True, BLACK)
        screen.blit(text_k, (1050, 50))

        # Run button
        pygame.draw.rect(screen, BLACK, (850, 150, 160, 50))
        screen.blit(text_run, (900, 150))

        # Random button
        pygame.draw.rect(screen, BLACK, (850, 250, 160, 50))
        screen.blit(text_random, (860, 250))

        # Algorithm button using scikit-learn
        pygame.draw.rect(screen, BLACK, (850, 450, 160, 50))
        screen.blit(text_algorithm, (860, 450))

        # Reset button scikit-learn
        pygame.draw.rect(screen, BLACK, (850, 550, 160, 50))
        screen.blit(text_reset, (860, 550))

        # Draw mouse position when mouse is in panel
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if 50 < mouse_x < 750 and 50 < mouse_y < 550:
            desc_mouse = f'({str(mouse_x - 50)}, {str(mouse_y - 50)})'
            text_mouse = font_sm.render(desc_mouse, True, BLACK)
            screen.blit(text_mouse, (mouse_x + 10, mouse_y))

        # End draw interface

        # Process device event (click, keyboard, ...)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Create point on panel
                if 50 < mouse_x < 750 and 50 < mouse_y < 550:
                    labels = []
                    point = [mouse_x - 50, mouse_y - 50]
                    points.append(point)

                # Change K button +
                if 850 < mouse_x < 900 and 50 < mouse_y < 100:
                    if k <= len(CLUSTER_COLORS):
                        k += 1

                # Change K button -
                if 950 < mouse_x < 1000 and 50 < mouse_y < 100:
                    if k > 0:
                        k -= 1

                # Press run button
                if 850 < mouse_x < 1000 and 150 < mouse_y < 200:
                    labels = []

                    if not clusters:
                        continue

                    # Assign lables
                    for point in points:
                        distances_to_cluster = []
                        for cluster in clusters:
                            d = distance(point, cluster)
                            distances_to_cluster.append(d)

                        # if distances_to_cluster:
                        min_d = min(distances_to_cluster)
                        label = distances_to_cluster.index(min_d)
                        labels.append(label)

                    # Update clusters
                    for i, cluster in enumerate(clusters):
                        sum_x = 0
                        sum_y = 0
                        count = 0

                        for j, point in enumerate(points):
                            if labels[j] == i:
                                sum_x += point[0]
                                sum_y += point[1]
                                count += 1

                        if count:
                            cluster[0] = int(sum_x / count)
                            cluster[1] = int(sum_y / count)

                # Press run button
                if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
                    clusters = []
                    labels = []

                    for _ in range(k):
                        cluster = [randint(10, 690), randint(10, 490)]
                        clusters.append(cluster)

                # Press Algorithm button
                if 850 < mouse_x < 1000 and 450 < mouse_y < 500:
                    try:
                        kmeans = KMeans(n_clusters=k).fit(points)
                        labels = list(kmeans.predict(points))
                        clusters = list(kmeans.cluster_centers_)
                    except Exception as e:
                        print(f'error: {str(e)}')

                # Press reset button
                if 850 < mouse_x < 1000 and 550 < mouse_y < 600:
                    k = 0
                    sse = 0
                    points = []
                    clusters = []
                    labels = []

        # Draw cluster
        for i, cluster in enumerate(clusters):
            cluster_point = (cluster[0] + 50, cluster[1] + 50)
            pygame.draw.circle(screen, CLUSTER_COLORS[i], cluster_point, 10)

        # Draw point
        for i, point in enumerate(points):
            element_point = (point[0] + 50, point[1] + 50)
            pygame.draw.circle(screen, BLACK, element_point, 6)

            if labels:
                point_label = labels[i]
                pygame.draw.circle(
                    screen, CLUSTER_COLORS[point_label], element_point, 5)
            else:
                pygame.draw.circle(screen, WHITE, element_point, 5)

        # Caculate and draw SEE - Sum of the squared error value
        sse = 0
        if clusters and labels:
            for point, label in zip(points, labels):
                sse += distance(point, clusters[label])

        text_sse = font_md.render('SSE = ' + str(int(sse)), True, BLACK)
        screen.blit(text_sse, (850, 350))

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
