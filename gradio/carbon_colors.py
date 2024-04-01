from __future__ import annotations


class Color:
    all = []

    def __init__(
        self,
        c50: str,
        c100: str,
        c200: str,
        c300: str,
        c400: str,
        c500: str,
        c600: str,
        c700: str,
        c800: str,
        c900: str,
        c950: str,
        name: str | None = None,
    ):
        self.c50 = c50
        self.c100 = c100
        self.c200 = c200
        self.c300 = c300
        self.c400 = c400
        self.c500 = c500
        self.c600 = c600
        self.c700 = c700
        self.c800 = c800
        self.c900 = c900
        self.c950 = c950
        self.name = name
        Color.all.append(self)

    def expand(self) -> list[str]:
        return [
            self.c50,
            self.c100,
            self.c200,
            self.c300,
            self.c400,
            self.c500,
            self.c600,
            self.c700,
            self.c800,
            self.c900,
            self.c950,
        ]


black = Color(
    name="black",
    c50="#000000",
    c100="#000000",
    c200="#000000",
    c300="#000000",
    c400="#000000",
    c500="#000000",
    c600="#000000",
    c700="#000000",
    c800="#000000",
    c900="#000000",
    c950="#000000",
)

blackHover = Color(
    name="blackHover",
    c50="#212121",
    c100="#212121",
    c200="#212121",
    c300="#212121",
    c400="#212121",
    c500="#212121",
    c600="#212121",
    c700="#212121",
    c800="#212121",
    c900="#212121",
    c950="#212121",
)

white = Color(
    name="white",
    c50="#ffffff",
    c100="#ffffff",
    c200="#ffffff",
    c300="#ffffff",
    c400="#ffffff",
    c500="#ffffff",
    c600="#ffffff",
    c700="#ffffff",
    c800="#ffffff",
    c900="#ffffff",
    c950="#ffffff",
)

whiteHover = Color(
    name="whiteHover",
    c50="#e8e8e8",
    c100="#e8e8e8",
    c200="#e8e8e8",
    c300="#e8e8e8",
    c400="#e8e8e8",
    c500="#e8e8e8",
    c600="#e8e8e8",
    c700="#e8e8e8",
    c800="#e8e8e8",
    c900="#e8e8e8",
    c950="#e8e8e8",
)

red = Color(
    name="red",
    c50="#fff1f1",
    c100="#ffd7d9",
    c200="#ffb3b8",
    c300="#ff8389",
    c400="#fa4d56",
    c500="#da1e28",
    c600="#a2191f",
    c700="#750e13",
    c800="#520408",
    c900="#2d0709",
    c950="#2d0709",
)

redHover = Color(
    name="redHover",
    c50="#540d11",
    c100="#66050a",
    c200="#921118",
    c300="#c21e25",
    c400="#b81922",
    c500="#ee0713",
    c600="#ff6168",
    c700="#ff99a0",
    c800="#ffc2c5",
    c900="#ffe0e0",
    c950="#ffe0e0",
)

blue = Color(
    name="blue",
    c50="#edf5ff",
    c100="#d0e2ff",
    c200="#a6c8ff",
    c300="#78a9ff",
    c400="#4589ff",
    c500="#0f62fe",
    c600="#0043ce",
    c700="#002d9c",
    c800="#001d6c",
    c900="#001141",
    c950="#001141",
)

blueHover = Color(
    name="blueHover",
    
    c50="#001f75",
    c100="#00258a",
    c200="#0039c7",
    c300="#0053ff",
    c400="#0050e6",
    c500="#1f70ff",
    c600="#5c97ff",
    c700="#8ab6ff",
    c800="#b8d3ff",
    c900="#dbebff",
    c950="#dbebff",
)


