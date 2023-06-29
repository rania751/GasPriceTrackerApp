from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd
import time
import datetime
from datetime import datetime
import pandas as pd
import csv 
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

generalinf_df = pd.DataFrame()
chrome_driver_path = 'C:/Users/r.kacem/Desktop/chromedriver_win32/chromedriver'
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service)
# abrir Google Chrome mediante chromedriver
driver.get('https://etherscan.io/')

while( True )  : 
    # while True:  
    #             try:
    #                 V1=True
                    
    #                 while V1:
    #                     try:
                
    #                             page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
    #                             V1=False  
    #                     except TimeoutException: 
    #                             time.sleep(0.5) 
    #                     else:
                            
    #                         break
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page2_element_click).perform()    


    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page2_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)
    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()      

    # while True:  
    #             try:
    #                 V2=True
    #                 while V2:
    #                     try: 
                           
                    
    #                         page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blocks"]')))
                            
    #                         v2=False
        
    #                     except TimeoutException: 
                                
    #                             time.sleep(0.5)        
    #                     else:
    #                                 break  
                
                    
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page3_element_click).perform()

    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page3_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)

    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()  
    
                
  


    v3=True
    while v3:
            try:
                  dernier_block_nbr_element0 = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="mCSB_1_container"]/div[1]/div[1]/div/div[2]/a '))) 
                  v3=False
            except TimeoutException: 
              time.sleep(0.5)         
            else:
                    break  
    try:
                
        dernier_block_nbr=(dernier_block_nbr_element0).text.strip()
    except:
                
        dernier_block_nbr=0 
    # v4=True
    # while v4:
    #         try:
                 
    #              principalpage_element= WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="masterHeader"]/nav/div/a/img[1]')))
    #              v4=False
    #         except TimeoutException: 
    #           time.sleep(0.5)                     
    #         else:
    #                 break      
    # actions = ActionChains(driver)
    # actions.move_to_element(principalpage_element).perform()
    # while True  :    
    #         try:    
               
    #             principalpage_element.click()
               
    #             break
    #         except ElementClickInterceptedException:
               
    #             time.sleep(0.5)
   
       
    v5=True  
    while v5:
            try:                                                                            
               trx_throughput_element  = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[2]/div[1]/div[2]/span') ))
               v5=False
            except TimeoutException: 
                break       


    try:
                trx_throughput=float(trx_throughput_element.text.strip().split("(")[1].strip().split("TPS")[0].strip())
    except:
                trx_throughput=0    


    # while True:  
    #             try:
    #                 V1=True
                    
    #                 while V1:
    #                     try:
                
    #                             page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
    #                             V1=False  
    #                     except TimeoutException: 
    #                             time.sleep(0.5) 
    #                     else:
                            
    #                         break
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page2_element_click).perform()    


    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page2_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)
    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()      

    # while True:  
    #             try:
    #                 V2=True
    #                 while V2:
    #                     try: 
                           
                    
    #                         page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blocks"]')))
                            
    #                         v2=False
        
    #                     except TimeoutException: 
                                
    #                             time.sleep(0.5)        
    #                     else:
    #                                 break  
                
                    
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page3_element_click).perform()

    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page3_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)

    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()  
            




    # v3=True
    # while v3:
    #         try:
    #               dernier_block_nbr_element1 = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="content"]/section[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]/a '))) 
    #               v3=False
    #         except TimeoutException: 
    #           time.sleep(0.5)         
    #         else:
    #                 break  
    # try:
                
    #     dernier_block_nbr1=(dernier_block_nbr_element1).text.strip()
    # except:
                
    #     dernier_block_nbr1=0     
    
    # while(dernier_block_nbr1==dernier_block_nbr):
    #          v3=True
    # while v3:
    #         try:
    #               dernier_block_nbr_element1 = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="content"]/section[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]/a '))) 
    #               v3=False
    #         except TimeoutException: 
    #           time.sleep(0.5)         
    #         else:
    #                 break  
    # try:
                
    #     dernier_block_nbr1=(dernier_block_nbr_element1).text.strip()
    # except:
                
    #     dernier_block_nbr1=0     
    
    # print(dernier_block_nbr1)
          
        
    # v6=True        
    # while v6:
    #         try:
    #            last_safe_block_element=  WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[2]/div[2]/div[3]/span')))
    #            v6=False
    #         except TimeoutException: 

    #             break     
    # v7=True          
    # while v7:
    #         try:
    #            last_finalized_block_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[2]/div[2]/div[2]/span')))
    #            v7=False
    #         except TimeoutException: 
    #             break                                  

    # try:
    #             trx_throughput=float(trx_throughput_element.text.strip().split("(")[1].strip().split("TPS")[0].strip())
    # except:
    #             trx_throughput=0    

    
    # try:
                
    #     last_safe_block=(last_safe_block_element.text.strip())
    # except:
                
    #     last_safe_block=0  

    # try:
                
    #     last_finalized_block=(last_finalized_block_element.text.strip())
    # except:
                
    #     last_finalized_block=0



    # while True:  
    #             try:
    #                 V8=True
                    
    #                 while V8:
    #                     try:
                
    #                             page_element_click= WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[1]/div[2]/div[2]/a')))
    #                             V8=False  
    #                     except TimeoutException: 
    #                             time.sleep(0.5) 
    #                     else:
                            
    #                         break
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page_element_click).perform()    


    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)
    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()          
    # v9=True
    # while v9:
    #     try:
    #                Market_cap_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_LitMarketCapTotal"]')))
    #                v9=False
   
    #     except TimeoutException: 

    #           break 
    # v10=True    
    # while v10:
    #         try:
    #               Total_ether_supply_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="content"]/div[2]/div[2]/div[2]/div/div/div[1]/div[1]/div/span')))
    #               v10=False
    #         except TimeoutException: 
 
    #           break 

    # try:

    #     Market_cap=(Market_cap_element.text.strip()[1:])
    # except:
                
    #     Market_cap=0
    # try:
                
    #     Total_ether_supply=(Total_ether_supply_element).text.strip()
    # except:
                
    #     Total_ether_supply=0
            

    # driver.back()

    while True:  
                try:
                    v11=True
                    while v11:
                        try:
                
                                page1_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[2]/div[1]/div[2]/a')))
                                V11=False  
                        except TimeoutException: 
                                time.sleep(0.5) 
                        else:
                            
                            break
                    actions = ActionChains(driver)
                    actions.move_to_element(page1_element_click).perform()    


                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page1_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)
                    break        
                except ElementClickInterceptedException:
                                        
                    driver.refresh()         
    
    # v12=True
    # while v12:       
    #     try:
    #          trnx_number_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '(//*[@id="ContentPlaceHolder1_divDataInfo"]/div/div[1]/span)')))
    #          v12=False
    #     except TimeoutException: 
  
    #          break     
            

    # try:
                
    #     trnx_number=(trnx_number_element).text.strip().split(">")[1].strip().split("transactions")[0].strip()
    # except:
                
    #     trnx_number=0
    # v13=True
    # while v13:       
    #     try:
    #         pending_trnx_1hour_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_divTxnStats"]/div[2]/a/div/div[2]/span[1]')))
    #         v13=False
    #     except TimeoutException: 
    #           time.sleep(0.5)      
    #     else:
    #          break  

    # try:
    #     pending_trnx_1hour_avg=pending_trnx_1hour_element.text.strip()
    # except:
    #     pending_trnx_1hour_avg=0


    # while True:  
    #             try:
    #                 v14=True
    #                 while v14:
    #                     try:
                
    #                             page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
    #                             V14=False  
    #                     except TimeoutException: 
    #                             time.sleep(0.5) 
    #                     else:
                            
    #                         break
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page2_element_click).perform()    


    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page2_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)
    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()      


    # while True:  
    #             try:
    #                 V15=True
    #                 while V15:
    #                     try: 
                           
                    
    #                         page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blocks"]')))
                            
    #                         v15=False
        
    #                     except TimeoutException: 
                                
    #                             time.sleep(0.5)        
    #                     else:
    #                                 break  
                
                    
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page3_element_click).perform()

    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page3_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)

    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()     
   
  

    # v16=True
    # while v16:
    #             try:

    #                 total_burnet_fee_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_divStats"]/div[4]/a/div/div[2]/span')))
    #                 v16=False
    #             except TimeoutException: 
    
    #                     break       
    # try:
                
    #     total_burnt_fee_ETH=(total_burnet_fee_element).text.strip().split("ETH")[0].strip()
    # except:
                
    #     total_burnt_fee_ETH=0
    # v17=True
    # while v17:
    #             try:
    #                 total_nbr_blocks_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="divDataInfo"]/div/div[1]/span[1]')))
    #                 v17=False
    #             except TimeoutException: 
  
    #                     break    
    # try:
                
    #     total_nbr_blocks=(total_nbr_blocks_element).text.strip().split("of")[1].strip().split("blocks")[0].strip()
    # except:
                
    #     total_nbr_blocks=0




    # while True:  
    #             try:
    #                 v18=True
    #                 while v18:
    #                     try:
                
    #                             page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
    #                             V18=False  
    #                     except TimeoutException: 
    #                             time.sleep(0.5) 
    #                     else:
                            
    #                         break
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page2_element_click).perform()    


    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page2_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)
    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()      


    # while True:  
    #             try:
    #                 V19=True
    #                 while V19:
    #                     try: 
                           
                    
    #                         page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI16"]')))
                            
    #                         V19=False
        
    #                     except TimeoutException: 
                                
    #                             time.sleep(0.5)        
    #                     else:
    #                                 break  
                
                    
    #                 actions = ActionChains(driver)
    #                 actions.move_to_element(page3_element_click).perform()

    #                 nbr=0
    #                 while True and (nbr<1) :    
    #                     try:    
                            
    #                         page3_element_click.click()
    #                         break
    #                     except ElementClickInterceptedException:
                            
                            
    #                         nbr=nbr+1
    #                         time.sleep(0.5)

    #                 break        
    #             except ElementClickInterceptedException:
                                        
    #                 driver.refresh()     

    # v20=True
    # while v20:
    #             try:
    #                 total_nbr_pending_trx_element = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ContentPlaceHolder1_divTopPage"]/div[1]/span[1]')))
    #                 v20=False
    #             except TimeoutException: 
    #                     break
    # try:
                
    #     total_nbr_pending_trx=(total_nbr_pending_trx_element).text.strip().split("of")[1].strip().split("pending")[0].strip()
    # except:
                
    #     total_nbr_pending_trx=0
    timestampp = time.time()
    formatted_time3 = datetime.fromtimestamp(timestampp).strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time3)
    print("dernier_block_nbr:",dernier_block_nbr)
    print("trx_throughput:",trx_throughput)
    # print("last_safe_block:",last_safe_block)
    # print("last_finalized_block:",last_finalized_block)
    # print("Market_cap:",Market_cap)
    # print("Total_ether_supply:",Total_ether_supply)
    # print("trnx_number:",trnx_number)
    # print("pending_trnx_1hour_avg:",pending_trnx_1hour_avg)
    # print("total_burnt_fee_ETH:",total_burnt_fee_ETH)    
    # print("total_nbr_blocks:",total_nbr_blocks)
    # print("total_nbr_pending_trx:",total_nbr_pending_trx) 
    # timestampp = time.time()
    # formatted_time3 = datetime.fromtimestamp(timestampp).strftime('%Y-%m-%d %H:%M:%S')
    # print(formatted_time3)
    my_dict={
                    
                     "dernier_block_nbr":dernier_block_nbr,
                    # "total_nbr_trx:":trnx_number,
                    # "nbre_pending_trnx:":total_nbr_pending_trx,
                    # "last_safe_block:":last_safe_block,
                    # "total_burnet_fees_ETH:":total_burnt_fee_ETH,
                    # "total_nbre_bloks:":total_nbr_blocks,
                    # "pending_trx_last_1H_AVG:":pending_trnx_1hour_avg,
                    "trx_throughput_TPS:":trx_throughput,
                    # "Market_cap:":Market_cap,
                    # "Total_ether_supply:":Total_ether_supply
                }
                
    df1=pd.DataFrame(my_dict,index=[0])
    generalinf_df= pd.concat([generalinf_df,df1])
    print(generalinf_df)
    generalinf_df.to_csv("THH35.csv", index=False)
    time.sleep(3)
    driver.back()
    driver.refresh()   