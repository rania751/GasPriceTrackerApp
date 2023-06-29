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
from selenium.webdriver.common.action_chains import ActionChains


gas_price_df = pd.DataFrame()
chrome_driver_path = 'C:/Users/r.kacem/Desktop/CHROME/chromedriver'
service = Service(executable_path=chrome_driver_path)


driver = webdriver.Chrome(service=service)
# abrir Google Chrome mediante chromedriver

driver.get("https://etherscan.io/")

while(True):
        while True:  
                try:
                    V1=True
                    
                    while V1:
                        try:
                
                                page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
                                V1=False  
                        except TimeoutException: 
                                time.sleep(0.5) 
                        else:
                            
                            break
                    actions = ActionChains(driver)
                    actions.move_to_element(page2_element_click).perform()    


                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page2_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)
                    break        
                except ElementClickInterceptedException:
                                        
                    driver.refresh()      


        while True:  
                try:
                    V2=True
                    while V2:
                        try: 
                            
                    
                            page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blocks"]')))
                            
                            v2=False
        
                        except TimeoutException: 
                                
                                time.sleep(0.5)        
                        else:
                                    break  
                
                    
                    actions = ActionChains(driver)
                    actions.move_to_element(page3_element_click).perform()

                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page3_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)

                    break        
                except ElementClickInterceptedException:
                                        
                    driver.refresh()      
        
                    
        V3=True
        while(V3):
                try:
                     dernier_block_nbr_element=driver.find_element(By.XPATH,'//*[@id="content"]/section[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]/a ') 
                     V3=False
                except NoSuchElementException:
                      time.sleep(0.5)       
                else:
                   break    
        try:
                    
            dernier_block_nbr=int((dernier_block_nbr_element).text.strip())
        except:
                    
            dernier_block_nbr=0    

        print("dernier_block_nbr:",dernier_block_nbr)
        V4=True
        while(V4):
                try:
                     principalpage_element= WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="masterHeader"]/nav/div/a/img[1]'))) 
                     V4=False
                except TimeoutException: 
                        time.sleep(0.5)         
                else:
                
                   break
        principalpage_element.click()
        V5=True
        while V5:
            try:
                
               Gas_price_dolar_element  =driver.find_element(By.XPATH,' //*[@id="ContentPlaceHolder1_mainboxes"]/div/div[2]/div[1]/div[3]/a/span')
               V5=False
            except NoSuchElementException:
                break       
        V6=True    
        while V6:
                    try:
                           
                          Med_Gas_price_gwei_element  = driver.find_element(By.XPATH,' //*[@id="ContentPlaceHolder1_mainboxes"]/div/div[2]/div[1]/div[3]/a')
                          V6=False
                    except NoSuchElementException:
                       
                        break   
        V7=True
        while V7:
                    try:
                          
                         Ether_price_element  =driver.find_element(By.XPATH,'//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[1]/div[1]/div[2]/a')
                         V7=False
                    except NoSuchElementException:
                        
                        break       
        V8=True            
        while V8:
                    try:
                          
                         Ether_price_avance_pourcentage_element= driver.find_element(By.XPATH,'//*[@id="ContentPlaceHolder1_mainboxes"]/div/div[1]/div[1]/div[2]/a/span[2]')
                         V8=False
                    except NoSuchElementException:
                        break 
         
        try:
              
              Gas_price_dolar=float(Gas_price_dolar_element.text.strip().split("$")[1].strip().split(")")[0].strip())
        except:
                
              Gas_price_dolar=0    


        try:
                
            Med_Gas_price_gwei=(Med_Gas_price_gwei_element.text.strip().split("G")[0].strip())
        except:
                
             Med_Gas_price_gwei=0      

        try:
                
            Ether_price=(Ether_price_element.text.strip().split("$")[1].strip().split("@")[0].strip())
        except:
                
            Ether_price=0  
        try:
                    
            Ether_price_avance_pourcentage=(Ether_price_avance_pourcentage_element.text.strip().split("(")[1].strip().split("%")[0].strip())
        except:
                
            Ether_price_avance_pourcentage=0             

        V9=True
        while(V9):
                try:
                     pagegastracker = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_services2"]')))
                     V9=False

                except TimeoutException: 
                        time.sleep(0.5)      
                else:
                
                   break     
        pagegastracker.click()
        V10=True
        while(V10):
                try:
                     pagegastrackerenter = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI19"]')))
                     V10=False

                except TimeoutException: 
                        time.sleep(0.5)      
                else:
                
                   break      
        pagegastrackerenter.click()
        V11=True
        while(V11):
                    try:
                        
                       low_gas_price_Gwei_element= driver.find_element(By.XPATH,'//*[@id="spanLowPrice"]')  
                       V11=False
                    except NoSuchElementException:
                       break  
        V12=True             
        while(V12):
                    try: 
                             
                            AVG_gas_price_Gwei_element= driver.find_element(By.XPATH,'//*[@id="spanAvgPrice"]') 
                            V12=False
                    except NoSuchElementException:
                      break   
        V13=True            
        while(V13):
                    try: 
                             
                            High_gas_price_Gwei_element= driver.find_element(By.XPATH,'//*[@id="spanHighPrice"] ')
                            V13=False
                    except NoSuchElementException:
                       break  
        V14=True             
        while(V14):
                    try:  
                             
                            priority_Low_gasprice_element = driver.find_element(By.XPATH,'//*[@id="spanLowPriorityAndBase"]') 
                            V14=False
                    except NoSuchElementException:
                       break   
        V15=True            
        while(V15):
                    try:  
                             
                            priority_Avg_gasprice_element=  driver.find_element(By.XPATH,' //*[@id="spanProposePriorityAndBase"]') 
                            V15=False
                    except NoSuchElementException:

                      break   
        V16=True            
        while(V16):
                    try: 
                             
                            priority_High_gasprice_element=  driver.find_element(By.XPATH,' //*[@id="spanHighPriorityAndBase"]') 
                            V16=False
                    except NoSuchElementException:
                       break   
        V17=True            
        while(V17):
                    try:
                             
                            low_gasprice_accept_time_element=  driver.find_element(By.XPATH,'//*[@id="divLowPrice"]/div[3]/span')
                            V17=False
                    except NoSuchElementException:
                        break   
        V18=True            
        while(V18):
                    try:
                          
                         Avg_gasprice_accept_time_element=  driver.find_element(By.XPATH,'//*[@id="divAvgPrice"]/div[3]/span')
                         V18=False
                    except NoSuchElementException:
                       break   
        V19=True            
        while(V19):
                    try:
                          
                         High_gasprice_accept_time_element=  driver.find_element(By.XPATH,' //*[@id="divHighPrice"]/div[3]/span')
                         V19=False
                    except NoSuchElementException:
                       break  
        V20=True             
        while(V20):
                    try:
                          
                         Low_gasprice_Dolar_element=  driver.find_element(By.XPATH,'//*[@id="divLowPrice"]/div[3]') 
                         V20=False
                    except NoSuchElementException:
                        break   
        V21=True            
        while(V21):
                    try:
                          
                         Avg_gasprice_Dolar_element=  driver.find_element(By.XPATH,'//*[@id="divAvgPrice"]/div[3]') 
                         V21=False
                    except NoSuchElementException:
                
                       break   
        V22=True            
        while(V22):
                    try:
                          
                         High_gasprice_Dolar_element=  driver.find_element(By.XPATH,'//*[@id="divHighPrice"]/div[3]')  
                         V22=False
                    except NoSuchElementException:

                       break                                                
        try:
            low_gas_price_Gwei=low_gas_price_Gwei_element.text
        except:
                    low_gas_price_Gwei=0  

        try:
             AVG_gas_price_Gwei=(AVG_gas_price_Gwei_element.text)
        except:
                    AVG_gas_price_Gwei=0 

        try:
              High_gas_price_Gwei=(High_gas_price_Gwei_element.text)
        except:
                    High_gas_price_Gwei=0           

        try:
                Base_Low_gasprice=(priority_Low_gasprice_element.text.strip().split("Base:")[1].strip().split("|")[0].strip())
        except:
                    priority_Low_gasprice=0      
        try:
               Base_Avg_gasprice=(priority_Avg_gasprice_element.text.strip().split("Base:")[1].strip().split("|")[0].strip())
        except:
                    priority_Avg_gasprice=0 
        try:
                Base_High_gasprice=(priority_High_gasprice_element.text.strip().split("Base:")[1].strip().split("|")[0].strip())
        except:
                    priority_High_gasprice=0        
        try:
              priority_Low_gasprice=(priority_Low_gasprice_element.text.strip().split("Priority:")[1].strip())
        except:
                    priority_Low_gasprice=0      
        try:
               priority_Avg_gasprice=(priority_Avg_gasprice_element.text.strip().split("Priority:")[1].strip())
        except:
                    priority_Avg_gasprice=0 
        try:
               priority_High_gasprice=(priority_High_gasprice_element.text.strip().split("Priority:")[1].strip())
        except:
                    priority_High_gasprice=0        
        try:
                 Low_gasprice_Dolar=(Low_gasprice_Dolar_element.text.strip().split("|")[0].strip()[1:])
        except:
                    priority_Low_gasprice_Dolar=0      
        try:
                 Avg_gasprice_Dolar=(Avg_gasprice_Dolar_element.text.strip().split("|")[0].strip()[1:])
        except:
                    priority_Avg_gasprice_Dolar=0 
        try:
                   High_gasprice_Dolar=(High_gasprice_Dolar_element.text.strip().split("|")[0].strip()[1:])
        except:
                    priority_High_gasprice_Dolar=0  
        try:
                   low_gasprice_accept_time=(low_gasprice_accept_time_element.text.strip())
        except:
                    low_gasprice_accept_time=0  
        try:
                   Avg_gasprice_accept_time=(Avg_gasprice_accept_time_element.text.strip())
        except:
                    Avg_gasprice_accept_time=0 
        try:
                   High_gasprice_accept_time=(High_gasprice_accept_time_element.text.strip())
        except:
                    High_gasprice_accept_time=0                  
        try:
                
                low_gasprice_accept_time_mins=(low_gasprice_accept_time.split("mins")[0].strip()[1:])

        except:
                low_gasprice_accept_time_mins=0   
                    

        if (":") in str(low_gasprice_accept_time):
                    low_gasprice_accept_time_sec=int(low_gasprice_accept_time.split(":")[1].strip().split("secs")[0].strip())
                    low_gasprice_accept_time=low_gasprice_accept_time_sec+(int(low_gasprice_accept_time_mins)*60)

        else:
                    if ("mins") not in low_gasprice_accept_time:
                        low_gasprice_accept_time_sec1=low_gasprice_accept_time.split("secs")[0].strip()[1:]
                        low_gasprice_accept_time=low_gasprice_accept_time_sec1

        try:
                
                Avggasprice_accept_time_mins=(Avg_gasprice_accept_time.split("mins")[0].strip()[1:])

        except:
                Avggasprice_accept_time_mins=0   
                    
        if (":") in Avg_gasprice_accept_time:
                    Avg_gasprice_accept_time_sec=int(Avg_gasprice_accept_time.split(":")[1].strip().split("secs")[0].strip())
                    Avg_gasprice_accept_time=Avg_gasprice_accept_time_sec+(int(Avggasprice_accept_time_mins)*60)

        else:
                    if ("mins") not in Avg_gasprice_accept_time:
                        Avg_gasprice_accept_time_sec1=Avg_gasprice_accept_time.split("secs")[0].strip()[1:]
                        Avg_gasprice_accept_time=Avg_gasprice_accept_time_sec1


        try:
                
                High_gasprice_accept_time_mins=int(High_gasprice_accept_time.split("mins")[0].strip()[1:])

        except:
                High_gasprice_accept_time_mins=0   
                    

            

        if (":") in High_gasprice_accept_time:
                    High_gasprice_accept_time_sec=int(High_gasprice_accept_time.split(":")[1].strip().split("secs")[0].strip())
                    High_gasprice_accept_time=High_gasprice_accept_time_sec+(int(High_gasprice_accept_time_mins)*60)

        else:
                    if ("mins") not in High_gasprice_accept_time:
                        High_gasprice_accept_time_sec1=High_gasprice_accept_time.split("secs")[0].strip()[1:]
                        High_gasprice_accept_time=High_gasprice_accept_time_sec1                

                                        

        print("Med_Gas_price_gwei:",Med_Gas_price_gwei)
        print("Gas_price_dolar:",Gas_price_dolar)
        print("Ether_price:",Ether_price)
        print("Ether_price_avance_pourcentage:",Ether_price_avance_pourcentage)
        print("low_gas_price_Gwei:",low_gas_price_Gwei)
        print("Avg_gas_price_Gwei:",AVG_gas_price_Gwei)
        print("High_gas_price_Gwei:",High_gas_price_Gwei)
        print("priority_Low_gasprice:",priority_Low_gasprice)
        print("priority_Avg_gasprice:",priority_Avg_gasprice)
        print("priority_High_gasprice:",priority_High_gasprice)
        print("Base_Low_gasprice:",Base_Low_gasprice)
        print("Base_Avg_gasprice:",Base_Avg_gasprice)
        print("Base_High_gasprice:",Base_High_gasprice)
        print("Low_gasprice_$:",Low_gasprice_Dolar)
        print("Avg_gasprice_$:",Avg_gasprice_Dolar)
        print("High_gasprice_$:",High_gasprice_Dolar)
        print("Low_gasprice_accept_time_sec:",low_gasprice_accept_time)
        print("Avg_gasprice_accept_time_sec:",Avg_gasprice_accept_time)
        print("High_gasprice_accept_time_sec:",High_gasprice_accept_time)



        while True:  
                try:
                    V23=True
                    while V23:
                        try:
                
                                page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
                                V1=False  
                        except TimeoutException: 
                                time.sleep(0.5) 
                        else:
                            
                            break
                    actions = ActionChains(driver)
                    actions.move_to_element(page2_element_click).perform()    


                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page2_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)
                    break        
                except ElementClickInterceptedException:
                                        
                    driver.refresh()  
        while True:  
                try:
                    V24=True
                    while V24:
                        try: 
                            
                    
                            page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blocks"]')))
                            
                            v2=False
        
                        except TimeoutException: 
                                
                                time.sleep(0.5)        
                        else:
                                    break  
                
                    
                    actions = ActionChains(driver)
                    actions.move_to_element(page3_element_click).perform()

                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page3_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)

                    break        
                except ElementClickInterceptedException:
                                        
                    driver.refresh()      

        V25=True
        while(V25):
                try:
                     
                    dernier_block_nbr_element1=  driver.find_element(By.XPATH,'//*[@id="content"]/section[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]/a ') 
                    V25=False
                except NoSuchElementException:    
                
                   break 
        try:
            
                    
            dernier_block_nbr1=int((dernier_block_nbr_element1).text.strip())
        except:
                    
            dernier_block_nbr1=0    

        print("dernier_block_nbr1:",dernier_block_nbr1)

        while(dernier_block_nbr1==dernier_block_nbr):

            while True:  
                try:
                    V26=True
                    while V26:
                        try:
                
                                page2_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blockchain"]')))
                                V26=False  
                        except TimeoutException: 
                                time.sleep(0.5) 
                        else:
                            
                            break
                    actions = ActionChains(driver)
                    actions.move_to_element(page2_element_click).perform()    


                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page2_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)
                    break        
                except ElementClickInterceptedException:
                            driver.refresh()  


            while True:  
                try:
                    V27=True
                    while V27:
                        try: 
                            
                    
                            page3_element_click = WebDriverWait(driver, 0,5).until(EC.presence_of_element_located((By.XPATH, '//*[@id="LI_blocks"]')))
                            
                            v27=False
        
                        except TimeoutException: 
                                
                                time.sleep(0.5)        
                        else:
                                    break  
                
                    
                    actions = ActionChains(driver)
                    actions.move_to_element(page3_element_click).perform()

                    nbr=0
                    while True and (nbr<1) :    
                        try:    
                            
                            page3_element_click.click()
                            break
                        except ElementClickInterceptedException:
                            
                            
                            nbr=nbr+1
                            time.sleep(0.5)

                    break        
                except ElementClickInterceptedException:
                                        
                    driver.refresh()      

                      
            V28=True
            while(V28):
                try:
                     
                    dernier_block_nbr_element1=  driver.find_element(By.XPATH,'//*[@id="content"]/section[2]/div[2]/div[2]/table/tbody/tr[1]/td[1]/a ') 
                    V28=False
                except NoSuchElementException:    
                   break 
            try:        
                dernier_block_nbr1=int((dernier_block_nbr_element1).text.strip())
            except:
                        
                dernier_block_nbr1=0    

            print("dernier_block_nbr:",dernier_block_nbr1)
           

        blockgasprice= dernier_block_nbr1+1  
        print(blockgasprice)
        print("fin")   
        timestampp = time.time()
        formatted_time3 = datetime.fromtimestamp(timestampp).strftime('%Y-%m-%d %H:%M:%S')
        print(formatted_time3)
        my_dict={
            "timeStamp":formatted_time3,
            "blockgasprice_number":blockgasprice,
            "Med_Gas_price_gwei:":Med_Gas_price_gwei,
            "Gas_price_dolar:":Gas_price_dolar,
            "Ether_price:":Ether_price,
            "Ether_price_avance_pourcentage:":Ether_price_avance_pourcentage,
            "low_gas_price_Gwei":low_gas_price_Gwei,
            "Avg_gas_price_Gwei":AVG_gas_price_Gwei,
            "High_gas_price_Gwei":High_gas_price_Gwei,
            "priority_Low_gasprice":priority_Low_gasprice,
            "priority_Avg_gasprice":priority_Avg_gasprice,
            "priority_High_gasprice":priority_High_gasprice,
            "Base_Low_gasprice":Base_Low_gasprice,
            "Base_Avg_gasprice":Base_Avg_gasprice,
            "Base_High_gasprice":Base_High_gasprice,
            "Low_gasprice_$":Low_gasprice_Dolar,
            "Avg_gasprice_$":Avg_gasprice_Dolar,
            "High_gasprice_$":High_gasprice_Dolar,
            "Low_gasprice_accept_time_sec":low_gasprice_accept_time,
            "Avg_gasprice_accept_time_sec":Avg_gasprice_accept_time,
            "High_gasprice_accept_time_sec":High_gasprice_accept_time,
        }
        df1=pd.DataFrame(my_dict,index=[0])
        gas_price_df = pd.concat([gas_price_df,df1])
        print(gas_price_df)
        gas_price_df.to_csv("gaspriceINF.csv", index=False)