class Math:
    def factorial(self, numb):
        if numb == 0:
            return 1
        else:
            return numb * self.factorial(numb - 1)

    def rectangleSurface(self, width, height):
        return width * height

    def circleSurface(self, radius):
        return 3.14 * radius ** 2

    def sum(self, *args):
        result = 0
        for arg in args:
            result += arg
        return result

    def multiply(self, *args):
        result = 1
        for arg in args:
            result *= arg
        return result

    def max(self, *args):
        result = args[0]
        for arg in args:
            if arg > result:
                result = arg
        return result

    def min(self, *args):
        result = args[0]
        for arg in args:
            if arg < result:
                result = arg
        return result