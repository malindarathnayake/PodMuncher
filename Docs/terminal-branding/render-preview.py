from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
root = Path.cwd()
font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 20)
label = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 17)
img = Image.new('RGB', (780, 430), '#141311')
draw = ImageDraw.Draw(img)
for index, style in enumerate(('classic', 'block')):
    path = root / 'ascii' / f'podmuncher-{style}-28.txt'
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines()
    assert max(map(len, lines)) == 28
    assert len(lines) == 14
    assert '\x1b' not in content
    if style == 'classic':
        assert content.isascii()
    draw.text((24 + index*390, 22), style + ' / 28 columns', font=label, fill='#FBF8F2')
    for row, line in enumerate(lines):
        draw.text((24 + index*390, 65 + row*23), line, font=font, fill='#FBF8F2')
    print(f'{path.name}: 28 columns x 14 rows, no ANSI escapes')
img.save(root / 'terminal-mascot-preview.png')
