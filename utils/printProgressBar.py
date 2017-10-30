import sys
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    string ='\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)
    sys.stdout.flush()
    sys.stdout.write(string)
    #print()

    # Print New Line on Complete
    # if iteration == total: 
    #     print()


# 
# Sample Usage
# 
if __name__ =='__main__':
    from time import sleep
    import sys
    
    # A List of Item
    items = list(range(0, 57))
    l = len(items)
    # Initial call to print 0% progress
    #flush()
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for j in xrange(5):
        #print ('this is {}'.format(j))
        print('')
        for i, item in enumerate(items):
            # Do stuff...
            sleep(0.1)
            sys.stdout.flush()
            accuracy = '{:.3f}%'.format(i/50.0)
            # Update Progress Bar
            printProgressBar(i + 1, l, prefix = 'Progress:', suffix = accuracy, length = 50)

# Sample Output
#Progress: ||||||||||||||||||||-----| 90.0% Complete
