import os
import sys
from pathlib import Path


if __name__ == '__main__':
    token = str(sys.argv[1])
    root_dir = Path(__file__).parent.parent
    file_path = os.path.join(root_dir, 'src', 'data',
                             'MONTEL_TOKEN.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(token)
    print(f'Stored token in token file: {token}')