from All_common_functions import *
# Initialize variables
bucket1 = 0
position1 = "on"
check_switch1 = "BUY"
check_switch5 = "SELL"
repair1 = "ON"
tr1 = "off"
p1s1 = 10
flb1 = 99
s11 = 0
p1s2 = 10
s21 = 0

# Load the data array from a file
d = np.load('Data_array.npy')

# Reset the 'bucket' row in the data array
bucket = d[0:1,1:]
buck_array = [bucket1]*8
buck_array = np.array(buck_array)
d[0:1,1:] = buck_array

# Reset the 'Position' row in the data array
Position =  d[1:2,1:]
position_array = [position1]*8
position_array = np.array(position_array)
d[1:2,1:] = position_array

# Reset the 'Check_switch' row in the data array
Check_switch= d[2:3,1:]
Check_switch_array = [check_switch1]*4 + [check_switch5]*4
Check_switch_array = np.array(Check_switch_array)
d[2:3,1:] = Check_switch_array

# Reset the 'repair' row in the data array
repair= d[3:4,1:]
repair_array = [repair1]*8
repair_array = np.array(repair_array)
d[3:4,1:] = repair_array

# Reset the 'tr' row in the data array
tr= d[4:5,1:]
tr_array = [tr1]*8
tr_array = np.array(tr_array)
d[4:5,1:] = tr_array

# Reset the 'ps1' row in the data array
ps1= d[5:6,1:]
ps1_array = [p1s1]*8
ps1_array = np.array(ps1_array)
d[5:6,1:] = ps1_array

# Reset the 'flb' row in the data array
flb= d[6:7,1:]
flb_array = [flb1]*8
flb_array = np.array(flb_array)
d[6:7,1:] = flb_array

# Reset the 's1' row in the data array
s1= d[7:8,1:]
s1_array = [s11]*8
s1_array = np.array(s1_array)
d[7:8,1:] = s1_array

# Reset the 'ps2' row in the data array
ps2= d[8:9,1:]
ps2_array = [p1s2]*8
ps2_array = np.array(ps2_array)
d[8:9,1:] = ps2_array

# Reset the 's2' row in the data array
s2= d[9:10,1:]
s2_array = [s21]*8
s2_array = np.array(s2_array)
d[9:10,1:] = s2_array

# Remove the old data array file
os.remove('Data_array.npy')

# Save the reset data array to a new file
np.save('Data_array', d)

# Print the reset data array
print(d)
