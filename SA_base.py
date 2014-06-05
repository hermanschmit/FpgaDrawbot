'''
SA_base
'''
import random,math

class SA_base(object):

    #FROZEN   =  100000
    #MAXMOVES = 1000000
    FROZEN    =  100000
    MAXMOVES  = 10000000
    
    def __init__(self,problemInst,t0=500.0,movesPerTemp=500,cool=0.9):
        self.inst = problemInst
        self.t0 = t0
        self.cool = cool
        self.movesPerTemp = movesPerTemp
        self.cost = self.inst.cost()
        self.resetStats()
        random.seed(25)

    def metropolis(self,delta,temp):
        if delta <= 0.0:
            return True
        rand = random.random()
        e = math.exp(-1.0*(delta/temp))
        if rand < e:
            return True
        return False

    def updateStats(self):
        delt = self.cost-self.m
        delt2 = delt*delt
        delt3 = delt2*delt
        self.m3 += delt3*self.tcount*(self.tcount-1)/((self.tcount+1)*(self.tcount+1)) - 3.0*delt*self.m2/(self.tcount+1)
        self.m2 += self.tcount*delt2/(self.tcount+1)
        self.m += delt/(self.tcount+1)
        self.sum += self.cost
        self.tcount += 1
        if delt > 0.0:
            self.hillclimb += 1

    def resetStats(self):
        self.tcount = 0
        self.m = 0
        self.m2 = 0
        self.m3 = 0
        self.sum = 0
        self.hillclimb = 0

    def costMean(self):
        if self.tcount == 0: return self.cost
        return self.sum/self.tcount

    def costStdDev(self):
        if self.tcount == 0: return 0.0
        return math.sqrt(self.m2/self.tcount)

    def costVar(self):
        if self.tcount == 0: return 0.0
        return self.m2/self.tcount

    def costSkewness(self):
        if self.tcount == 0 or self.m2 == 0.0: return 0.0
        return math.sqrt(self.tcount)*self.m3/math.pow(self.m2,1.5)

    def commitMove(self,delta):
        self.inst.commitMove()
        self.cost += delta
        self.updateStats()

    def newTemp(self,temp):
        self.resetStats()
        return temp * self.cool

    def optimize(self):
        moves = 0
        costList = []
        temp = self.t0
        frozenCount = 0
        lastCost = 0
        frozenMove = 0
        while (frozenCount == 0 or (moves-frozenMove) < self.FROZEN) and moves < self.MAXMOVES:
            for j in xrange(0, self.movesPerTemp):
                moves += 1
                delta = self.inst.genMove()
                if self.metropolis(delta,temp):
                    self.commitMove(delta)
            cost = self.inst.cost()
            if abs(cost - self.inst.currentCost) > 1e-7:
                print "weird: ", cost, self.inst.currentCost
            if lastCost == cost:
                if frozenCount == 0: frozenMove = moves
                frozenCount += 1
            else:
                frozenCount = 0
            temp = self.newTemp(temp)
            costList.append(moves)
            costList.append(self.costMean())
            lastCost = cost
            print temp, cost
        return costList    

if __name__ == '__main__':
    import TravSales
    inst = TravSales.TravSales()
    base = SA_base(inst,500,10000,0.9)
    print base.optimize()

    base.resetStats()
    for i in xrange(0,100):
        base.cost = 7.0
        base.updateStats()
    for i in xrange(5,10):
        base.cost = i*1.0
        base.updateStats()
    print "Number: ",base.tcount
    print "Mean: ",base.costMean()
    print "Dev: ",base.costStdDev()
    print "Skew: ",base.costSkewness()

    base.resetStats()
    for i in xrange(0,1000000):
        base.cost = random.random()*6 + random.random()*6 + random.random()*6
        base.updateStats()
    print "Number: ",base.tcount
    print "Mean: ",base.costMean()
    print "Dev: ",base.costStdDev()
    print "Skew: ",base.costSkewness()


        


    

    
        
