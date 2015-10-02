__author__ = 'herman'
from unittest import TestCase

import Sketchy


class Sketchy_BotTransformRaw1(TestCase):
    def setUp(self):
        pass


class TestSketchy_transform1(Sketchy_BotTransformRaw1):
    def runTest(self):
        c = (700, 700)
        x, y = Sketchy.botTransformReverse(c, 0, 0, 1000)
        print x, y
