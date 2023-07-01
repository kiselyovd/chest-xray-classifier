from torchvision import transforms

# Аугментатор для обучающих изображений
augmentator = transforms.Compose([
    # Размытие по Гауссу
    transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 2)),
    # Случайный горизонтальный переворот
    transforms.RandomHorizontalFlip(p=0.5),
    # Случайный вертикальный переворот
    transforms.RandomVerticalFlip(p=0.5),
    # Случайный контраст
    transforms.RandomAutocontrast(),
    transforms.RandomChoice([
        # Эластичная трансформация
        transforms.ElasticTransform(),
        # Случайная перспектива
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    ]),
    # Случайный поворот
    transforms.RandomRotation(degrees=(0, 90)),
    # Стирание случайной прямоугольной области
    transforms.RandomErasing(p=0.3),
])
