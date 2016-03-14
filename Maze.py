
import Segments
import numpy as np

class Maze:

    K0 = 0.1     # [0.1;0.3]
    K1 = 0.15    # [1.5*K0; 2.5*K0]
    D = 10       # dimensional adjustment?
    KMIN = 0.2
    KMAX = 1.2
    Ff = 0.005   # [0.005; 0.3]
    Fb = 0.1     # [0; 0.2]
    Fa = 5.      # [0; 10]
    NMIN = 1     # [1,2]
    SIGMA_LJ = 1. # Is this K0 * D == R0

    def delta(self,i):
        d = self.seg.segmentList[0][i]
        v = self.imin[d[0], d[1]]
        assert v >= 0
        assert v < 256
        return float(v+1)/256.


    def brownian(self):
        mean = [0., 0.]
        cov = [[1.,0.], [0.,1.]]
        size = len(self.seg.segmentList[0])
        x,y = np.random.multivariate_normal(mean, cov, size)
        z = zip(x,y)
        brown = []
        for i, zi in enumerate(z):
            n = np.array(zi)
            n = np.multiply(n, self.Fb*self.delta(i)*self.D)
            brown.append(n)
        return np.array(brown)

    def faring(self):
        fare = []
        for i in xrange(1,
                        len(self.seg.segmentList[0])-2):
            pim1 = self.seg.segmentList[0][i-1]
            pi   = self.seg.segmentList[0][i]
            pip1 = self.seg.segmentList[0][i+1]

            f = self.Ff * (((pim1*self.delta(i-1) + pip1*self.delta(i+1))/(self.delta(i-1)+self.delta(i+1))) - pi)
            fare.append(f)
        return np.array(fare)

    def LennardJones(self,r):
        d = self.SIGMA_LJ/r
        return d**12 - d**6

    def attract_repel_slow(self):
        '''
        This is the brute force version
        Returns:
        attract repel vector
        '''
        returnList = []
        R1 = self.K1*self.K0*self.D
        for i in xrange(0,
                        len(self.seg.segmentList[0])):
            fi = np.array([0.,0.])
            delta_pi = self.delta(i)
            pi = self.seg.segmentList[0][i]
            for j in xrange(0,
                            len(self.seg.segmentList[0])-1):
                indexDistance = max(abs(j-i),abs(j+1-i))
                if indexDistance > self.NMIN:
                    pi2xij,xij = self.seg.distABtoP(j,j+1,i)
                    deltaj = self.delta(j)
                    deltajp1 = self.delta(j+1)
                    deltamin = min(deltaj,deltajp1,delta_pi)
                    if pi2xij < deltamin*R1:
                        fij = (pi-xij)/pi2xij
                        fij *= self.LennardJones(pi2xij/(delta_pi*self.D))
                        fi += fij
            returnList.append(fi)
        return np.array(returnList)

    def __init__(self, image_matrix, white=1):
        """
        :param image_matrix:
        """

        self.imin = image_matrix
         # whiten
        self.imin /= white
        self.imin += 255 - (255 // white)

        # TODO: do we want to quantize the image?

        # Initial segment
        segList1 = [(0., 0.)]
        segListEnd = tuple([x-1 for x in self.imin.shape])
        segList1.append(segListEnd)
        self.seg = Segments.Segments()
        self.seg.append(segList1)
        self.seg.scale(1.0) # fix the types. Hygiene

        # Main optimize loop
        # keep running until stopping criteria met
        while True:
            # compute force on each node
            brownian = self.brownian()
            attract_repel = self.attract_repel_slow()
            fairing = self.faring()

            # move each node

            # resampling: segment/desegment

            # stopping criteria
            if True:
                break

