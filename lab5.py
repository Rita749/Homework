import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

class XORNeuralNetworkDemo:
    def __init__(self):
        self.X = None
        self.y = None
        self.centers = None
        self.original_colors = None
        self.final_colors = None

    def generate_xor_data(self, num_points_per_class=50):
        """Генерация данных для задачи XOR"""
        np.random.seed(42)
        # Центры для XOR задачи
        self.centers = [
            [0, 0],  # Множество 0: (0,0) → Класс 0
            [0, 1],  # Множество 1: (0,1) → Класс 1
            [1, 0],  # Множество 2: (1,0) → Класс 1
            [1, 1]   # Множество 3: (1,1) → Класс 0
        ]
        
        # Исходные цвета для 4 множеств
        self.original_colors = ['blue', 'red', 'green', 'orange']
        # Итоговые цвета для 2 различных классов
        self.final_colors = ['lightblue', 'lightcoral']
        
        X = []
        y = []
        
        for class_idx, center in enumerate(self.centers):
            for i in range(num_points_per_class):
                # Добавляем небольшой шум вокруг центров
                x1 = center[0] + random.uniform(-0.2, 0.2)
                x2 = center[1] + random.uniform(-0.2, 0.2)
                X.append([x1, x2])
                # Преобразуем в бинарные классы: множества 0 и 3 → класс 0, множества 1 и 2 → класс 1
                binary_class = 0 if class_idx in [0, 3] else 1
                y.append(binary_class)
        
        self.X = np.array(X)
        self.y = np.array(y)
        return self.X, self.y, self.original_colors, self.final_colors

    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        """Производная сигмоидной функции"""
        return x * (1 - x)

    def two_layer_nn_solution(self, class_0_mask, class_1_mask):
        """Двухслойная нейронная сеть для решения XOR"""
        # Инициализация весов
        np.random.seed(42)
        input_size = 2
        hidden_size = 4
        output_size = 1
        
        # Веса для скрытого слоя
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(1, hidden_size)
        
        # Веса для выходного слоя
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(1, output_size)
        
        # Параметры обучения
        learning_rate = 0.5
        epochs = 10000
        
        X = self.X
        y = self.y.reshape(-1, 1)
        
        # Обучение сети
        for epoch in range(epochs):
            # Прямое распространение
            hidden_input = np.dot(X, W1) + b1
            hidden_output = self.sigmoid(hidden_input)
            
            final_input = np.dot(hidden_output, W2) + b2
            final_output = self.sigmoid(final_input)
            
            # Обратное распространение
            error = y - final_output
            
            # Градиенты для выходного слоя
            d_output = error * self.sigmoid_derivative(final_output)
            
            # Градиенты для скрытого слоя
            error_hidden = d_output.dot(W2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)
            
            # Обновление весов
            W2 += hidden_output.T.dot(d_output) * learning_rate
            b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            W1 += X.T.dot(d_hidden) * learning_rate
            b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
            
            if epoch % 2000 == 0:
                loss = np.mean(np.square(error))
                print(f"Эпоха {epoch}, Ошибка: {loss:.6f}")
        
        # Тестирование на всех комбинациях XOR
        test_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        print("\nТестирование на всех комбинациях XOR:")
        for i, point in enumerate(test_points):
            hidden_input = np.dot(point, W1) + b1
            hidden_output = self.sigmoid(hidden_input)
            final_input = np.dot(hidden_output, W2) + b2
            final_output = self.sigmoid(final_input)
            
            # Исправление: извлекаем скалярное значение из массива
            output_value = final_output[0][0] if final_output.ndim > 1 else final_output[0]
            prediction = 1 if output_value > 0.5 else 0
            print(f"Вход: {point} → Выход: {output_value:.4f} → Предсказание: {prediction}")
        
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    def step_function(self, x):
        """Ступенчатая функция активации"""
        return 1 if x >= 0 else 0

    def manual_xor_calculation(self):
        """Ручной расчет XOR с пошаговым объяснением"""
        print("\n=== РУЧНОЙ РАСЧЕТ XOR ===")
        
        # Тестовые точки
        test_points = [[0, 0], [0, 1], [1, 0], [1, 1]]
        
        for point in test_points:
            x1, x2 = point
            
            print(f"\n--- Расчет для ({x1}, {x2}) ---")
            
            # Первый слой: OR и NAND
            or_result = self.step_function(x1 + x2 - 0.5)
            nand_result = self.step_function(-x1 - x2 + 1.5)
            
            print(f"OR нейрон: {x1} + {x2} - 0.5 = {x1 + x2 - 0.5:.1f} → {or_result}")
            print(f"NAND нейрон: -{x1} - {x2} + 1.5 = {-x1 - x2 + 1.5:.1f} → {nand_result}")
            
            # Второй слой: AND(OR, NAND)
            xor_result = self.step_function(or_result + nand_result - 1.5)
            
            print(f"AND(OR, NAND): {or_result} + {nand_result} - 1.5 = {or_result + nand_result - 1.5:.1f} → {xor_result}")
            print(f"XOR({x1}, {x2}) = {xor_result}")

    def visualize_results(self, network):
        """Визуализация результатов"""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Исходные данные
        ax1 = fig.add_subplot(gs[0, 0])
        for i, center in enumerate(self.centers):
            mask = (self.X[:, 0] >= center[0]-0.3) & (self.X[:, 0] <= center[0]+0.3) & \
                   (self.X[:, 1] >= center[1]-0.3) & (self.X[:, 1] <= center[1]+0.3)
            ax1.scatter(self.X[mask, 0], self.X[mask, 1], 
                       c=self.original_colors[i], label=f'Множество {i}')
        ax1.set_title('1. Исходные 4 множества данных')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Финальные классы
        ax2 = fig.add_subplot(gs[0, 1])
        class_0_mask = self.y == 0
        class_1_mask = self.y == 1
        ax2.scatter(self.X[class_0_mask, 0], self.X[class_0_mask, 1], 
                   c=self.final_colors[0], label='Класс 0', alpha=0.7)
        ax2.scatter(self.X[class_1_mask, 0], self.X[class_1_mask, 1], 
                   c=self.final_colors[1], label='Класс 1', alpha=0.7)
        ax2.set_title('2. Финальные 2 класса (XOR)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Архитектура сети
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        # Текст с объяснением
        explanation_text = """
        АРХИТЕКТУРА РЕШЕНИЯ XOR:
        
        1. ПЕРВЫЙ СЛОЙ создает промежуточные признаки:
           • OR нейрон: активируется когда x1 ИЛИ x2 = 1
           • NAND нейрон: активируется когда НЕ(x1 И x2)
        
        2. ВТОРОЙ СЛОЙ комбинирует:
           • AND(OR, NAND) = XOR
        
        3. РЕЗУЛЬТАТ для всех комбинаций:
           • (0,0): OR=0, NAND=1 → AND(0,1)=0 
           • (0,1): OR=1, NAND=1 → AND(1,1)=1  
           • (1,0): OR=1, NAND=1 → AND(1,1)=1 
           • (1,1): OR=1, NAND=0 → AND(1,0)=0 
        """
        
        ax3.text(0.1, 0.9, explanation_text, fontsize=12, va='top', 
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def main():
    """Основная демонстрационная программа для XOR"""
    demo = XORNeuralNetworkDemo()
    
    print("ДЕМОНСТРАЦИЯ: НЕЙРОННАЯ СЕТЬ ДЛЯ ЗАДАЧИ XOR")
    print("=" * 70)
    
    # Генерируем данные
    print("Генерация данных XOR...")
    X, y, original_colors, final_colors = demo.generate_xor_data(50)
    
    # Показываем первоначальную постановку
    print("\n1. Постановка задачи XOR:")
    class_0_mask = y == 0
    class_1_mask = y == 1
    print(f"   Класс 0: {np.sum(class_0_mask)} точек")
    print(f"   Класс 1: {np.sum(class_1_mask)} точек")
    
    input("\nНажмите Enter для продолжения...")
    
    # Демонстрация решения
    print("\n2. Решение XOR с помощью двухслойной нейронной сети...")
    network = demo.two_layer_nn_solution(class_0_mask, class_1_mask)
    
    print("\n" + "=" * 70)
    print("РЕЗЮМЕ: КАК СЕТЬ РЕШАЕТ XOR")
    print("=" * 70)
    
    # Ручной расчет
    demo.manual_xor_calculation()
    
    # Визуализация
    print("\nСоздание визуализации...")
    demo.visualize_results(network)
    
    print("\nДемонстрация завершена!")

if __name__ == "__main__":

    main()
