import numpy as np
import random, copy, math

def hyp(ax,ay,bx,by):
    xdiff = ax - bx
    ydiff = ay - by
    return math.hypot(xdiff,ydiff)

class CannyTSP(object):
    def __init__(self, segmentList):
        self.num = len(segmentList)
        self.OrderedList = segmentList[:]
        self.candidateMove = [0,0]
        self.currentCost = self.cost()
        self.currentDelta = 0

    # def hyp_int(i1,i2):
    #     a = self.OrderedList[i1]
    #     b = self.OrderedList[i2]
    #     return hyp(*(a+b))

    def genMove(self):
        self.candidateMove = [0,0]
        def validMove():
            if self.candidateMove[0] == self.candidateMove[1]:
                return False
            # TODO fix this limitation
            if abs(abs(self.candidateMove[0]) - abs(self.candidateMove[1])) == 1:
                return False
            if self.candidateMove[0] == 0:
                return False
            if self.candidateMove[1] == 0:
                return False
            # TODO fix this limitation
            if abs(self.candidateMove[0]) == self.num-1:
                return False
            if abs(self.candidateMove[1]) == self.num-1:
                return False
            return True

        while not validMove():
            self.candidateMove = [random.randint(1-self.num,self.num-1),
                                  random.randint(1-self.num,self.num-1)]
        a = abs(self.candidateMove[0])
        b = abs(self.candidateMove[1])
        o1 = a-1
        o2 = a+1
        o3 = b-1
        o4 = b+1
        delta = 0

        #if a == b:
        #    pass
        if b-a == 1:
            # o1 --> a[0] --> a[-1] --> b[0] --> b[-1] --> o4
            delta -= hyp(*(self.OrderedList[a][0]  + self.OrderedList[o1][-1]))
            delta -= hyp(*(self.OrderedList[a][-1] + self.OrderedList[b][0]))
            delta -= hyp(*(self.OrderedList[b][-1] + self.OrderedList[o4][0]))
            if self.candidateMove[0] < 0:
                delta += hyp(*(self.OrderedList[a][0]  + self.OrderedList[o4][0]))
                a_idx = -1
            else:
                delta += hyp(*(self.OrderedList[a][-1]  + self.OrderedList[o4][0]))
                a_idx = 0
            if self.candidateMove[1] < 0:
                delta += hyp(*(self.OrderedList[b][-1]  + self.OrderedList[o1][-1]))
                b_idx = 0
            else:
                delta += hyp(*(self.OrderedList[b][0]  +  self.OrderedList[o1][-1]))
                b_idx = -1
            delta += hyp(*(self.OrderedList[a][a_idx]  +  self.OrderedList[b][b_idx]))
        elif a-b == 1:
            # o3 --> b[0] --> b[-1] --> a[0] --> a[-1] --> o2
            delta -= hyp(*(self.OrderedList[b][0]  + self.OrderedList[o3][-1]))
            delta -= hyp(*(self.OrderedList[b][-1] + self.OrderedList[a][0]))
            delta -= hyp(*(self.OrderedList[a][-1]+ self.OrderedList[o2][0]))
            if self.candidateMove[0] < 0:
                delta += hyp(*(self.OrderedList[a][-1]  + self.OrderedList[o3][-1]))
                a_idx = 0
            else:
                delta += hyp(*(self.OrderedList[a][0]  + self.OrderedList[o3][-1]))
                a_idx = -1
            if self.candidateMove[1] < 0:
                delta += hyp(*(self.OrderedList[b][0]  + self.OrderedList[o2][0]))
                b_idx = -1
            else:
                delta += hyp(*(self.OrderedList[b][-1]  + self.OrderedList[o2][0]))
                b_idx = 0
            delta += hyp(*(self.OrderedList[a][a_idx]  +  self.OrderedList[b][b_idx]))
        else:
            # o1 --> a[0] --> a[-1] --> o2
            # o3 --> b[0] --> b[-1] --> o4

            delta -= hyp(*(self.OrderedList[a][0] + self.OrderedList[o1][-1]))
            delta -= hyp(*(self.OrderedList[a][-1]+ self.OrderedList[o2][0]))
            delta -= hyp(*(self.OrderedList[b][0] + self.OrderedList[o3][-1]))
            delta -= hyp(*(self.OrderedList[b][-1]+ self.OrderedList[o4][0]))
            if self.candidateMove[0] < 0:
                delta += hyp(*self.OrderedList[a][0]+self.OrderedList[o4][0])
                delta += hyp(*self.OrderedList[a][-1]+self.OrderedList[o3][-1])
            else:
                delta += hyp(*self.OrderedList[a][-1]+self.OrderedList[o4][0])
                delta += hyp(*self.OrderedList[a][0]+self.OrderedList[o3][-1])
            if self.candidateMove[1] < 0:
                delta += hyp(*self.OrderedList[b][0]+self.OrderedList[o2][0])
                delta += hyp(*self.OrderedList[b][-1]+self.OrderedList[o1][-1])
            else:
                delta += hyp(*self.OrderedList[b][-1]+self.OrderedList[o2][0])
                delta += hyp(*self.OrderedList[b][0]+self.OrderedList[o1][-1])


        #orig_cost = self.cost()
        #self.move()
        #delta2 = self.cost() - orig_cost
        # # undo the move
        #self.move()
        #self.move()
        #self.move()
        #if delta != delta2:
        #    print "weird: ",delta,delta2


        #         + hyp(*(self.OrderedList[a]  +self.OrderedList[b+1]))
        #         - hyp(*(self.OrderedList[a-1]+self.OrderedList[a]))
        #         - hyp(*(self.OrderedList[a+1]+self.OrderedList[a]))
        #         - hyp(*(self.OrderedList[b-1]+self.OrderedList[b]))
        #         - hyp(*(self.OrderedList[b+1]+self.OrderedList[b])))

        self.currentDelta = delta

        return delta

    def cost(self):
        accumCost = 0
        for i in xrange(0,self.num-1):
            accumCost += hyp(*(self.OrderedList[i][-1]+self.OrderedList[i+1][0]))
        return accumCost

    def move(self):
        if self.candidateMove[0] < 0:
            self.OrderedList[abs(self.candidateMove[0])].reverse()
        if self.candidateMove[1] < 0:
            self.OrderedList[abs(self.candidateMove[1])].reverse()

        (self.OrderedList[abs(self.candidateMove[0])],
         self.OrderedList[abs(self.candidateMove[1])]) = \
        (self.OrderedList[abs(self.candidateMove[1])],
         self.OrderedList[abs(self.candidateMove[0])])

    def mergeIfNeighbor(self,a,b):
        if b-a != 1:
            return
        if a < 0 or b >= self.num:
            return
        a_e = self.OrderedList[a][-1]
        b_b = self.OrderedList[b][0]
        dist = hyp(*a_e+b_b)
        if dist < 2:
            self.OrderedList[a] = self.OrderedList[a] + self.OrderedList[b]
            del self.OrderedList[b]
            self.currentDelta -= dist
            self.num -= 1

    def commitMove(self):
        self.move()
        a = abs(self.candidateMove[1])
        b = abs(self.candidateMove[0])
        self.mergeIfNeighbor(a-1,a)
        self.mergeIfNeighbor(a,a+1)
        self.mergeIfNeighbor(b-1,b)
        self.mergeIfNeighbor(b,b+1)

        self.candidateMove = [0,0]
        self.currentCost += self.currentDelta

