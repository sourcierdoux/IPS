from locate_user_ips import *
from locate_user_ips import mv_positioning, plot_positions



def tracking_full():

    shared_data={}
    #threading.Thread(target=locate_ble, args=(shared_data,), daemon=True).start()  
    thread_mv=threading.Thread(target=mv_positioning, args=(shared_data,False),daemon=True)
    thread_ble=threading.Thread(target=locate_ble, args=(shared_data,),daemon=True)
    thread_mv.start()
    thread_ble.start()

    thread_ble.join()
    print("Hi ! IM DONE")
    if 'MV' in shared_data and 'BLE' in shared_data:
        person_to_follow=match_id(shared_data)
    thread_follow=threading.Thread(target=follow, args=(person_to_follow,), daemon=True)
    thread_follow.start()
    plot_positions(shared_data=shared_data)
        #send_to_fan(client_socket2=client_socket2,persons=persons)

    
    #client_socket2.close()

def main():
    #mv_positioning(shared_data=None,plot=True)
    #locate_ble(shared_data=None,plot=True,n_measurement=50,current_position=True)
    tracking_full()
if __name__=="__main__":
    main()