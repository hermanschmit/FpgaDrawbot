import numpy
import math

class LSystem():

    def __init__(self,axiom=None,rules=None,angle=90):
        self.axiom = axiom
        self.rules = rules
        self.angle = math.radians(angle)
        self.working = None

    def segment(self,initialpt=[0.0,0.0],initialheading=0.0,d=1.0):
        pt = numpy.array(initialpt)
        heading = math.radians(initialheading)
        seg = list()
        seg.append(numpy.array(pt))
        for char in self.working:
            if char == 'A' or char == 'B' or char == 'F':
                pt += d*numpy.array([math.cos(heading),math.sin(heading)])
                seg.append(numpy.array(pt))
            elif char == '+':
                heading += self.angle
            elif char == '-':
                heading -= self.angle
        return numpy.array(seg)

    def iterate(self,repetitions=1):
        if self.axiom == None or self.rules == None:
            return
        self.working = self.axiom
        for repeat in range(0,repetitions):
            newpath=""
            i = 0
            while i<len(self.working):
                found=False
                for lhs,rhs in self.rules:
                    if lhs == self.working[i:i+len(lhs)]:
                        newpath+=rhs
                        i+=len(lhs)
                        found=True
                        break
                if not found:
                    newpath+=self.working[i:i+1]
                    i+=1
            self.working = newpath
        return self.working
