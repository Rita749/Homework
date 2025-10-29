import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.errors = []
    
    def activation(self, x):
        """Функция активации: 1 если x >= 0, -1 если x < 0"""
        return np.where(x >= 0, 1, -1)
    
    def fit(self, X, y):
        """Обучение персептрона"""
        n_samples, n_features = X.shape
        
        # Инициализация весов случайными значениями
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            total_error = 0
            for idx, x_i in enumerate(X):
                # Прямое распространение
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                
                # Обновление весов (правило обучения персептрона)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
                
                # Суммирование ошибок
                total_error += (y[idx] - y_pred) ** 2
            
            self.errors.append(total_error)
            
            # Если ошибка равна 0, останавливаем обучение
            if total_error == 0:
                break
    
    def predict(self, X):
        """Предсказание класса для новых данных"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# Генерация данных
def generate_data():
    # Первоначальная точка для первого класса
    center1 = np.array([2, 3])
    # Генерация 12 точек вокруг center1 с нормальным распределением
    points1 = center1 + np.random.randn(12, 2) * 0.8
    
    # Находим точку, максимально удаленную от center1
    distances = np.linalg.norm(points1 - center1, axis=1)
    farthest_point_idx = np.argmax(distances)
    center2 = points1[farthest_point_idx]
    
    # Генерация 12 точек вокруг center2
    points2 = center2 + np.random.randn(12, 2) * 0.8
    
    return points1, points2, center1, center2

# Визуализация данных и разделяющей линии
def plot_results(points1, points2, perceptron, center1, center2):
    plt.figure(figsize=(15, 5))
    
    # График 1: Исходные данные
    plt.subplot(1, 3, 1)
    plt.scatter(points1[:, 0], points1[:, 1], c='blue', label='Класс C1', alpha=0.7)
    plt.scatter(points2[:, 0], points2[:, 1], c='red', label='Класс C2', alpha=0.7)
    plt.scatter(center1[0], center1[1], c='darkblue', marker='*', s=200, label='Центр C1')
    plt.scatter(center2[0], center2[1], c='darkred', marker='*', s=200, label='Центр C2')
    plt.title('Исходные данные')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Разделяющая линия
    plt.subplot(1, 3, 2)
    plt.scatter(points1[:, 0], points1[:, 1], c='blue', label='Класс C1', alpha=0.7)
    plt.scatter(points2[:, 0], points2[:, 1], c='red', label='Класс C2', alpha=0.7)
    
    # Построение разделяющей линии
    x1_min, x1_max = min(points1[:, 0].min(), points2[:, 0].min()), max(points1[:, 0].max(), points2[:, 0].max())
    x1_range = np.linspace(x1_min - 0.5, x1_max + 0.5, 100)
    
    if perceptron.weights[1] != 0:  # Избегаем деления на ноль
        # x2 = - (w1*x1 + bias) / w2
        x2_range = -(perceptron.weights[0] * x1_range + perceptron.bias) / perceptron.weights[1]
        plt.plot(x1_range, x2_range, 'g-', linewidth=2, label='Разделяющая линия')
    
    plt.title('Классификация персептроном')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Ошибка обучения
    plt.subplot(1, 3, 3)
    plt.plot(perceptron.errors)
    plt.title('Ошибка обучения')
    plt.xlabel('Итерация')
    plt.ylabel('Суммарная ошибка')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Основная программа
def main():
    # Генерация данных
    points1, points2, center1, center2 = generate_data()
    
    print(f"Центр класса C1: {center1}")
    print(f"Центр класса C2: {center2}")
    print(f"Расстояние между центрами: {np.linalg.norm(center1 - center2):.2f}")
    
    # Проверка пересечения множеств
    min_dist_between_classes = np.min([np.linalg.norm(p1 - p2) for p1 in points1 for p2 in points2])
    print(f"Минимальное расстояние между точками разных классов: {min_dist_between_classes:.2f}")
    
    # Подготовка данных для обучения
    X = np.vstack([points1, points2])  # Объединяем все точки
    # Добавляем фиктивный признак x3 = 1 для смещения (bias)
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    
    # Метки классов: 1 для C1, -1 для C2
    y = np.array([1] * len(points1) + [-1] * len(points2))
    
    # Создание и обучение персептрона
    perceptron = Perceptron(learning_rate=0.1, n_iters=100)
    perceptron.fit(X_with_bias, y)
    
    # Предсказание на обучающих данных
    y_pred = perceptron.predict(X_with_bias)
    
    # Оценка точности
    accuracy = np.mean(y == y_pred)
    print(f"\nТочность классификации: {accuracy * 100:.2f}%")
    
    # Вывод параметров модели
    print(f"\nОбученные веса:")
    print(f"w1 = {perceptron.weights[0]:.4f}")
    print(f"w2 = {perceptron.weights[1]:.4f}")
    print(f"Смещение (bias) = {perceptron.bias:.4f}")
    
    # Коэффициент k для разделяющей линии
    if perceptron.weights[1] != 0:
        k = -perceptron.weights[0] / perceptron.weights[1]
        print(f"Коэффициент k разделяющей линии: {k:.4f}")
    
    # Визуализация
    plot_results(points1, points2, perceptron, center1, center2)
    
    # Дополнительная проверка
    print("\nПроверка разделяющей функции:")
    print("Уравнение разделяющей линии:")
    print(f"{perceptron.weights[0]:.4f}*x1 + {perceptron.weights[1]:.4f}*x2 + {perceptron.bias:.4f} = 0")

if __name__ == "__main__":
    main()