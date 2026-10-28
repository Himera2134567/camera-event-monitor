import os
import sys

def process_labels_dir(labels_dir):
    for fname in os.listdir(labels_dir):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(labels_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            new_lines.append('0 ' + ' '.join(parts[1:]) + '\n')
        # сохраняем обратно
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'Обработан: {path}')

def main(root_dir):
    if not os.path.isdir(root_dir):
        print(f'Ошибка: директория "{root_dir}" не найдена.')
        sys.exit(1)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).lower() == 'labels':
            process_labels_dir(dirpath)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Использование: python converter.py <корневая_папка_с_train_valid>')
        sys.exit(1)
    main(sys.argv[1])
