import paramiko

 
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
 
server = "192.168.0.212"  
user = "libedev"  
pwd = "liberabit!@" 
 
cli.connect(server, port=22, username=user, password=pwd)
stdin, stdout, stderr = cli.exec_command("python3 mute/mute-hero/NEW_Data_preprocessing/Sound_deeplearning/Sound_deeplearning2/Deep_Test2.py")
lines = stderr.readlines()
print(''.join(lines))
if (lines == ['True\n']):
    # airplane is OK
    #GPIO.output(17,GPIO.HIGH)
    print("hello")

else
    #GPIO LOW... 
    #GPIO.output(17,GPIO.LOW)


cli.close()