from enum import Enum


class ShowType(Enum):
    DEFAULT = 0
    HIGHLIGHT = 1
    UNDERLINE = 4
    BLINKING = 5
    REVERSE = 7
    HIDE = 8


class Foreground(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37


class Background(Enum):
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47

def color_font(text: str, foreground: Foreground = None, background: Background = None, show_type: ShowType = None) -> str:
    """
    给文本添加颜色和样式

    :param text: 文本内容
    :param foreground: 前景色
    :param background: 背景色
    :param show_type: 显示类型
    :return: 带有颜色和样式的文本
    """
    codes = []
    if foreground is not None:
        codes.append(str(foreground.value))
    if background is not None:
        codes.append(str(background.value))
    if show_type is not None:
        codes.append(str(show_type.value))

    if codes:
        return f"\033[{';'.join(codes)}m{text}\033[0m"
    else:
        return text
