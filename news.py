import webbrowser

urls = [
    "https://www.example1.com",
    "https://www.example2.com",
    "https://www.example3.com"
]

for url in urls:
    webbrowser.open_new_tab(url)
