import random
import logging
import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

today_date = str(datetime.datetime.now()).split(" ")[0]
current_path = os.path.dirname(os.path.realpath(__file__))

start_time = datetime.datetime.now()
info_handler = logging.FileHandler(os.path.join(current_path, 'pb1.4-'+str(start_time))+'-info.log')
info_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)

logger.addHandler(info_handler)

# targetFunction f = 2x1 - x2 + 3
THRESHOLD = random.randint(-100,100)

## To achieve
W_FINAL = [random.randint(-100,100),random.randint(-100,100)]


## Generating Data Set
def generate_input_space(DATA_SIZE):
    x = []
    for i in range(0,DATA_SIZE):
        param1 = random.randint(0,30)
        param2 = random.randint(0,30)
        x.append([param1,param2])

    logger.info(x)
    return x

## Generating output space y
def generate_output_space(x):
    global W_FINAL
    y = []
    for i in range(0, len(x)):
        val = W_FINAL[0] * x[i][0] + W_FINAL[1] * x[i][1] + THRESHOLD
        if val > 0:
            y.append(1)
        elif val < 0:
            y.append(-1)
        else:
            logger.info("Random data set included one data point (%d,%d) which lies on the line of actual target function" %(x[i][0],x[i][1]))
            y.append("ignore: invalid data point")

    logger.info(y)
    return y

## @input_param w: weights ( list of length=2)
## @input_param x: input data space (list of length = 2)
## @Output : returns 1 if classification is correct
##           returns -1 if classification is incorrect
##           returns 0 if point lies on Perceptron line
def hypothesis_function(w,x):
    if (w[0] * x[0] + w[1] * x[1] + THRESHOLD) > 0:
        return 1
    elif (w[0] * x[0] + w[1] * x[1] + THRESHOLD) < 0:
        return -1
    else:
        print("Point x1:%d,x2:%d lies on perceptron line with weights:%d,%d" %(x[0],x[1],w[0],w[1]))
        return 0

## Part of Perceptron Learning Algorithm (PLA)
## @input_param w: weights (list of length = 2)
## @input param x: misclassified input (list of length =2)
## @input_param y: known decision output (integer)
## @Output: returns updated weights (list of length = 2)
def update_function(w,x,y_out):
    return [w[0]+y_out*x[0],w[1]+y_out*x[1]]

def is_misclassified(x1,x2,y_out,w1,w2):
    global THRESHOLD

    if y_out == "ignore: invalid data point":
        return False

    if (w1 * x1 + w2 * x2 + THRESHOLD)*y_out < 0:
        return True
    else:
        return False


## Part of Perceptron Learning Algorithm (PLA)
## @input_param w: weights (list of length = 2)
## @Output: missclassified data point
def find_misclassified(w,x,y):
    global THRESHOLD
    count = 5 # number of times to randomly choose a point in given space
              # if all points are classified then go through all input space

    while(count>0):
        i = random.randint(-1,len(x)-1)
        if is_misclassified(x[i][0],x[i][1],y[i],w[0],w[1]):
            return [x[i],y[i]]
        count -=1

    for i in range(0,len(x)):
        if is_misclassified(x[i][0],x[i][1],y[i],w[0],w[1]):
            return [x[i],y[i]]
    logger.info("Hurray! No Misclassified data point exists")
    print("Hurray! No Misclassified data point exists")
    return None


def plotPerceptron(w,x,y,run_count):
    global W_FINAL
    sns.set()
    all_x1 = []
    for i in range(0,len(x)):
        all_x1.append(x[i][0])

    all_y1 = [(-float(w[0])/float(w[1])*i -THRESHOLD/float(w[1])) for i in all_x1]
    plt.plot(all_x1, all_y1, color='orange', linewidth=1.0, label='Hypothesis g()')
    target_y = [(-1 * float(W_FINAL[0]) / float(W_FINAL[1])) * i - THRESHOLD / float(W_FINAL[1]) for i in all_x1]
    plt.plot(all_x1, target_y, color='b', linewidth=1.0, label='Target f()')
    for i in range(0, len(x)):
        if y[i] == "ignore: invalid data point":
            continue
        if y[i] < 0:
            plt.plot(x[i][0], x[i][1], 'ro',label='Negative Region')
        else:
            plt.plot(x[i][0], x[i][1], 'go',label='Positive Region')

    plt.xlabel("x1 [Run #%d, Data Size: %d]" % (run_count, len(x)))
    plt.ylabel("x2")

    # line_plotted1 = mpatches.Patch(color='blue', label='Target f()')
    # line_plotted2 = mpatches.Patch(color='orange', label='Hypothesis g()')
    # plt.legend(handles=[line_plotted1,line_plotted2])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),frameon=True)
    plt.show()

def createPlot(x,y,run_count):
    global W_FINAL
    all_x1 = []
    sns.set()
    for i in range(0,len(x)):
        if y[i] == "ignore: invalid data point":
            continue
        all_x1.append(x[i][0])
        if y[i] < 0:
            plt.plot(x[i][0],x[i][1],'ro',label='Negative Region')
        else:
            plt.plot(x[i][0],x[i][1],'go',label='Positive Region')

    plt.xlabel("x1 [Run #%d, Data Size: %d]"%(run_count,len(x)))
    plt.ylabel("x2")
    target_y = [(-1*float(W_FINAL[0])/float(W_FINAL[1]))*i-THRESHOLD/float(W_FINAL[1]) for i in all_x1]
    plt.plot(all_x1,target_y,color='b', linewidth=2.0,label='Target f()')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),frameon=True)
    plt.show()

## PLA knows only about input x, output y and threshold
def PLA(x,y,run_count):
    global THRESHOLD, start_time

    # Let initial assumed weights be 8 & 5 respectively
    w = [random.randint(-100,100),random.randint(-100,100)]
    PLA_ITERATION_t = 1
    while True:
        print("Run %d: - PLA iteration: %d" %(run_count, PLA_ITERATION_t))
        logger.info("Run %d: - PLA iteration: %d" %(run_count, PLA_ITERATION_t))
        misclassified_pt = find_misclassified(w,x,y)
        if misclassified_pt is not None:
            w = update_function(w,misclassified_pt[0],misclassified_pt[1])
            logger.info("Weights udpated....")
            logger.info(w)

        else:
            print("******Solution found!**********")
            logger.info("******Solution found!**********")
            logger.info("PLA Weights are below")
            logger.info(w)
            logger.info("Actual Weights were: ")
            logger.info(W_FINAL)
            logger.info("Hypothesis Function g: (%d x1)+(%d x2)+%d" % (w[0], w[1], THRESHOLD))
            logger.info("Target Function t: (%d x1)+(%d x2)+%d" % (W_FINAL[0], W_FINAL[1], THRESHOLD))
            plotPerceptron(w,x,y,run_count)
            pla_slope = (float(w[0])/float(w[1]))
            actual_slope = float(W_FINAL[0])/float(W_FINAL[1])
            error_percent = abs((pla_slope-actual_slope)/actual_slope)
            logger.info("Error in slope: %s" %(error_percent))
            logger.info("time taken : %s" % (str(datetime.datetime.now() - start_time)))
            logger.info("Solution found in iteration: %d" %(PLA_ITERATION_t))
            return
        PLA_ITERATION_t += 1
    logger.info("This should never print")
    return


## Calls PLA with different DATA_SIZE
def runScript(DATA_SIZE,run_count):
    logger.info("**** RUN COUNT: %d *******"%(run_count))
    logger.info("Generating input space for data size: %d" %(DATA_SIZE))
    x = generate_input_space(DATA_SIZE)
    logger.info("Input space Successfully Generated")
    logger.info("Generating Output space")
    y = generate_output_space(x)
    logger.info("Output Space Successfully Generated")
    logger.info("Plotting scatter plot with target function f")
    createPlot(x,y,run_count)
    logger.info("Commencing PLA for Data size=%d" %(DATA_SIZE))
    print("Commencing PLA for Data size=%d"%(DATA_SIZE))
    PLA(x,y,run_count)





if __name__ == "__main__":
    logger.info("*********************************")
    print("*********************************")
    logger.info("Solution Prob 1.4: --by Vimanyu Aggarwal 011006125")
    logger.info("*********************************")
    print("*********************************")
    DATA_SIZE = [20,20,100,1000]
    run_count = 1
    for i in range(0,4):
        runScript(DATA_SIZE[i],run_count)
        run_count +=1
    print("Please see logs for results!")
    logger.info("Shutting down!")
