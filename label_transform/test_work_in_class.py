import multiprocessing as mp
import pathos.pools as pp
def work(test_obj,input_list):
		#print input_list
		print test_obj.num_process
class test_works_in_class:
	def __init__(self):
		self.num_process = 10
	def call_back(self):
		print 'call back done'
	def work(self,input_list):
		print input_list
	def start(self):
		print('start')
		t_list = [i for i in range(20)]
		steps = len(t_list)/self.num_process
		pool = mp.Pool(processes=self.num_process)
		#p = pp.ProcessPool(self.num_process)
		for i in range(self.num_process):
			input_list = t_list[steps*i:steps*(i+1)] if i <self.num_process-1 else t_list[steps*i:]
			pool.apply(work, args=(self,input_list,))
if __name__ =='__main__':
	#print ('start')
	tw = test_works_in_class()
	tw.start()

