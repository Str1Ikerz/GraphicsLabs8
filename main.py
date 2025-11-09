import math
import struct
from typing import List, Tuple, Callable


class BMPImage:
    """Класс для работы с BMP изображениями"""

    def __init__(self, width: int, height: int, pixels: List[List[Tuple[int, int, int]]] = None):
        self.width = width
        self.height = height
        if pixels is None:
            self.pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
        else:
            self.pixels = pixels

    def save(self, filename: str):
        """Сохранение изображения в BMP формат"""
        row_size = (self.width * 3 + 3) & ~3
        image_size = row_size * self.height
        file_size = 54 + image_size

        header = bytearray(54)
        struct.pack_into('<2sI', header, 0, b'BM', file_size)
        struct.pack_into('<HHI', header, 6, 0, 0, 54)
        struct.pack_into('<IiiHHIIiiII', header, 14,
                         40, self.width, self.height, 1, 24, 0, image_size, 0, 0, 0, 0)

        data = bytearray()
        for y in range(self.height - 1, -1, -1):
            row = bytearray()
            for x in range(self.width):
                r, g, b = self.pixels[y][x]
                row.extend([b, g, r])
            while len(row) < row_size:
                row.append(0)
            data.extend(row)

        with open(filename, 'wb') as f:
            f.write(header)
            f.write(data)

    def copy(self):
        """Создание копии изображения"""
        copied_pixels = [row[:] for row in self.pixels]
        return BMPImage(self.width, self.height, copied_pixels)


class ImageTransformer:
    """Класс для преобразования изображений"""

    @staticmethod
    def nearest_interpolation(src: BMPImage, x: float, y: float) -> Tuple[int, int, int]:
        """Интерполяция по ближайшему соседу"""
        ix = min(src.width - 1, max(0, round(x)))
        iy = min(src.height - 1, max(0, round(y)))
        return src.pixels[iy][ix]

    @staticmethod
    def bilinear_interpolation(src: BMPImage, x: float, y: float) -> Tuple[int, int, int]:
        """Билинейная интерполяция"""
        x = max(0, min(src.width - 1, x))
        y = max(0, min(src.height - 1, y))

        x1, y1 = int(x), int(y)
        x2, y2 = min(src.width - 1, x1 + 1), min(src.height - 1, y1 + 1)

        a = x - x1
        b = y - y1

        def interpolate(c00, c10, c01, c11):
            return (1 - a) * (1 - b) * c00 + a * (1 - b) * c10 + (1 - a) * b * c01 + a * b * c11

        r = interpolate(src.pixels[y1][x1][0], src.pixels[y1][x2][0],
                        src.pixels[y2][x1][0], src.pixels[y2][x2][0])
        g = interpolate(src.pixels[y1][x1][1], src.pixels[y1][x2][1],
                        src.pixels[y2][x1][1], src.pixels[y2][x2][1])
        b_val = interpolate(src.pixels[y1][x1][2], src.pixels[y1][x2][2],
                            src.pixels[y2][x1][2], src.pixels[y2][x2][2])

        return (int(r), int(g), int(b_val))

    @staticmethod
    def bicubic_interpolation(src: BMPImage, x: float, y: float) -> Tuple[int, int, int]:
        """Бикубическая интерполяция"""

        def get_pixel(ix, iy):
            ix = max(0, min(src.width - 1, ix))
            iy = max(0, min(src.height - 1, iy))
            return src.pixels[iy][ix]

        x_int, y_int = int(x), int(y)
        a, b = x - x_int, y - y_int

        # Вычисление коэффициентов для бикубической интерполяции
        c1 = (a - 1) * (a - 2) * (a + 1) * (b - 1) * (b - 2) * (b + 1) / 4
        c2 = -a * (a - 2) * (a + 1) * (b - 1) * (b - 2) * (b + 1) / 4
        c3 = b * (a - 1) * (a - 2) * (a + 1) * (b - 2) * (b + 1) / 4
        c4 = a * b * (a - 2) * (a + 1) * (b - 2) * (b + 1) / 4
        c5 = -a * (a - 1) * (a - 2) * (b - 1) * (b - 2) * (b + 1) / 12
        c6 = -b * (a - 1) * (a - 2) * (a + 1) * (b - 1) * (b - 2) / 12
        c7 = a * b * (a - 1) * (a - 2) * (b - 2) * (b + 1) / 12
        c8 = a * b * (a - 2) * (a + 1) * (b - 1) * (b - 2) / 12
        c9 = a * (a - 1) * (a + 1) * (b - 1) * (b - 2) * (b + 1) / 12
        c10 = b * (a - 1) * (a - 2) * (a + 1) * (b - 1) * (b + 1) / 12
        c11 = a * b * (a - 1) * (a - 2) * (b - 1) * (b - 2) / 36
        c12 = -a * b * (a - 1) * (a + 1) * (b - 2) * (b + 1) / 12
        c13 = -a * b * (a - 2) * (a + 1) * (b - 1) * (b + 1) / 12
        c14 = -a * b * (a - 1) * (a + 1) * (b - 1) * (b - 2) / 36
        c15 = -a * b * (a - 1) * (a - 2) * (b - 1) * (b + 1) / 36
        c16 = a * b * (a - 1) * (a + 1) * (b - 1) * (b + 1) / 36

        pixels = [
            get_pixel(x_int, y_int), get_pixel(x_int, y_int + 1),
            get_pixel(x_int + 1, y_int), get_pixel(x_int + 1, y_int + 1),
            get_pixel(x_int, y_int - 1), get_pixel(x_int - 1, y_int),
            get_pixel(x_int + 1, y_int - 1), get_pixel(x_int - 1, y_int + 1),
            get_pixel(x_int, y_int + 2), get_pixel(x_int + 2, y_int),
            get_pixel(x_int - 1, y_int - 1), get_pixel(x_int + 1, y_int + 2),
            get_pixel(x_int + 2, y_int + 1), get_pixel(x_int - 1, y_int + 2),
            get_pixel(x_int + 2, y_int - 1), get_pixel(x_int + 2, y_int + 2)
        ]

        coefficients = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]

        r = sum(p[0] * coef for p, coef in zip(pixels, coefficients))
        g = sum(p[1] * coef for p, coef in zip(pixels, coefficients))
        b_val = sum(p[2] * coef for p, coef in zip(pixels, coefficients))

        return (max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b_val))))

    @staticmethod
    def get_inverse_matrix(matrix: List[List[float]]) -> List[List[float]]:
        """Вычисление обратной матрицы для аффинного преобразования"""
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        det = a * e - b * d

        if abs(det) < 1e-10:
            return [[1, 0, 0], [0, 1, 0]]

        a_inv = e / det
        b_inv = -b / det
        c_inv = (b * f - e * c) / det
        d_inv = -d / det
        e_inv = a / det
        f_inv = (d * c - a * f) / det

        return [[a_inv, b_inv, c_inv], [d_inv, e_inv, f_inv]]

    def affine_transform(self, image: BMPImage, transform_matrix: List[List[float]],
                         interpolation_func: Callable, use_center: bool = True) -> BMPImage:
        """Применение аффинного преобразования"""
        inv_matrix = self.get_inverse_matrix(transform_matrix)
        result = image.copy()

        for y in range(image.height):
            for x in range(image.width):
                if use_center:
                    x_centered = x - image.width / 2
                    y_centered = y - image.height / 2

                    src_x_centered = (inv_matrix[0][0] * x_centered +
                                      inv_matrix[0][1] * y_centered +
                                      inv_matrix[0][2])
                    src_y_centered = (inv_matrix[1][0] * x_centered +
                                      inv_matrix[1][1] * y_centered +
                                      inv_matrix[1][2])

                    src_x = src_x_centered + image.width / 2
                    src_y = src_y_centered + image.height / 2
                else:
                    src_x = inv_matrix[0][0] * x + inv_matrix[0][1] * y + inv_matrix[0][2]
                    src_y = inv_matrix[1][0] * x + inv_matrix[1][1] * y + inv_matrix[1][2]

                if 0 <= src_x < image.width and 0 <= src_y < image.height:
                    result.pixels[y][x] = interpolation_func(image, src_x, src_y)
                else:
                    result.pixels[y][x] = (0, 0, 0)  # Черный цвет для пикселей вне исходного изображения

        return result


class TransformationFactory:
    """Фабрика для создания матриц преобразований"""

    @staticmethod
    def scaling(sx: float, sy: float) -> List[List[float]]:
        """Матрица масштабирования"""
        return [[sx, 0, 0], [0, sy, 0]]

    @staticmethod
    def rotation(angle: float) -> List[List[float]]:
        """Матрица поворота"""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]]

    @staticmethod
    def skew_x(skew: float) -> List[List[float]]:
        """Матрица скоса по X"""
        return [[1, skew, 0], [0, 1, 0]]

    @staticmethod
    def skew_y(skew: float) -> List[List[float]]:
        """Матрица скоса по Y"""
        return [[1, 0, 0], [skew, 1, 0]]


class TestImageGenerator:
    """Генератор тестовых изображений"""

    @staticmethod
    def create_complex_test() -> BMPImage:
        """Создание сложного тестового изображения"""
        width, height = 200, 200
        image = BMPImage(width, height)

        # Красный квадрат
        for y in range(50, 150):
            for x in range(50, 150):
                image.pixels[y][x] = (255, 0, 0)

        # Зеленый треугольник
        for y in range(30, 80):
            for x in range(30, 30 + (y - 30)):
                image.pixels[y][x] = (0, 255, 0)

        # Синий круг
        center_x, center_y = 100, 100
        radius = 20
        for y in range(center_y - radius, center_y + radius):
            for x in range(center_x - radius, center_x + radius):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    image.pixels[y][x] = (0, 0, 255)

        # Серая сетка
        for y in range(0, height, 20):
            for x in range(width):
                if image.pixels[y][x] == (0, 0, 0):
                    image.pixels[y][x] = (150, 150, 150)

        for x in range(0, width, 20):
            for y in range(height):
                if image.pixels[y][x] == (0, 0, 0):
                    image.pixels[y][x] = (150, 150, 150)

        return image


def main():
    """Основная функция программы"""
    print("=== Программа преобразования изображений ===\n")

    # Создание тестового изображения
    print("Создание тестового изображения...")
    generator = TestImageGenerator()
    test_image = generator.create_complex_test()

    try:
        test_image.save('test_input.bmp')
        print("✓ Тестовое изображение создано: test_input.bmp")
    except Exception as e:
        print(f"✗ Ошибка при сохранении тестового изображения: {e}")
        return

    # Инициализация трансформера
    transformer = ImageTransformer()

    # Определение преобразований
    transformations = [
        ('scale_1.5x', TransformationFactory.scaling(1.5, 1.5)),
        ('scale_0.7x', TransformationFactory.scaling(0.7, 0.7)),
        ('rotate_45', TransformationFactory.rotation(45)),
        ('rotate_-20', TransformationFactory.rotation(-20)),
        ('skew_x_0.3', TransformationFactory.skew_x(0.3)),
        ('skew_y_0.3', TransformationFactory.skew_y(0.3)),
    ]

    # Определение методов интерполяции
    interpolation_methods = [
        ('nearest', transformer.nearest_interpolation),
        ('bilinear', transformer.bilinear_interpolation),
        ('bicubic', transformer.bicubic_interpolation)
    ]

    # Применение преобразований
    print("\nПрименение преобразований...")
    successful_transforms = 0

    for trans_name, matrix in transformations:
        print(f"\nПреобразование: {trans_name}")

        for interp_name, interp_func in interpolation_methods:
            try:
                result = transformer.affine_transform(test_image, matrix, interp_func)
                filename = f'result_{trans_name}_{interp_name}.bmp'
                result.save(filename)
                print(f"  ✓ Создан файл: {filename}")
                successful_transforms += 1
            except Exception as e:
                print(f"  ✗ Ошибка при создании {filename}: {e}")

    # Итоговая статистика
    print(f"\n=== Итог ===")
    print(f"Создано файлов: {successful_transforms + 1} (1 исходный + {successful_transforms} преобразований)")
    print("Все файлы сохранены в текущей директории")


if __name__ == "__main__":
    main()