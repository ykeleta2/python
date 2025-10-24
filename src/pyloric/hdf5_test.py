import numpy as np
import tables  as tb


#out = np.random.randint(0,200,(100,100))
#my_file=tb.open_file("test_hdf.h5",mode="w",title="Test File")
#grp=my_file.create_group(my_file.root,"data")
#my_file.create_array(grp,"my_array",out,"my matrix data")
#my_file.close()
print("==============================================")

read_file=tb.open_file("src/pyloric/data/output_300x300.h5",mode="r")
mat_from_file=read_file.root.data.out_array.read()

read_file.close()
print(mat_from_file)