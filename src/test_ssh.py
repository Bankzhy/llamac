
def run():
    with open("outputs/test.txt", encoding="utf-8") as f:
        content = "test"
        f.write(content)

if __name__ == '__main__':
    run()