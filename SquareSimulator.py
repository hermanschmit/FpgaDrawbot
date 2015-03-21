__author__ = 'herman'

def main(scale, center=(0.5,0.5), incr=0.0001):


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], (sys.argv[2], sys.argv[3]), sys.argv[4])
    elif len(sys.argv) == 4:
        main(sys.argv[1], (sys.argv[2], sys.argv[3]))
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print "Error: unknown usage"