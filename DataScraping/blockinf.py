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
from selenium.webdriver.chrome.options import Options
options = webdriver.ChromeOptions()
generalinf_df = pd.DataFrame()
chrome_driver_path = 'C:/Users/r.kacem/Desktop/chromedriver_win32/chromedriver'
service = Service(executable_path=chrome_driver_path)

driver = webdriver.Chrome(service=service,options=options)

# abrir Google Chrome mediante chromedriver

lista=[16942905, 16942906, 16942908, 16942909, 16942911, 16942912, 16942913, 16942914, 16942916, 16942917, 16942918, 16942919, 16942920, 16942922, 16942923, 16942924, 16942926, 16942928, 16942930, 16942931, 16942932, 16942933, 16942934, 16942935, 16942936, 16942937, 16942938, 16942939, 16942940, 16942942, 16942943, 16942944, 16942945, 16942946, 16942948, 16942949, 16942951, 16942952, 16942954, 16942955, 16942956, 16942957, 16942958, 16942959, 16942960, 16942961, 16942962, 16942963, 16942964, 16942965, 16942966, 16942967, 16942968, 16942969, 16942970, 16942971, 16942972, 16942973, 16942974, 16942975, 16942976, 16942977, 16942978, 16942979, 16942980, 16942982, 16942984, 16942985, 16942986, 16942987, 16942989, 16942990, 16942992, 16942993, 16942994, 16942995, 16942996, 16942997, 16942999, 16943000, 16943001, 16943003, 16943005, 16943006, 16943008, 16943009, 16943010, 16943011, 16943012, 16943014, 16943015, 16943016, 16943017, 16943018, 16943020, 16943021, 16943023, 16943024, 16943025, 16943026, 16943035, 16943038, 16943039, 16943040, 16943042, 16943043, 16943044, 16943045, 16943047, 16943048, 16943049, 16943050, 16943051, 16943052, 16943053, 16943055, 16943057, 16943058, 16943059, 16943060, 16943062, 16943063, 16943064, 16943065, 16943066, 16943068, 16943069, 16943077, 16943079, 16945173, 16945174, 16945175, 16945176, 16945177, 16945178, 16945179, 16945180, 16945181, 16945182, 16945183, 16945184, 16945185, 16945186, 16945188, 16945189, 16945190, 16945191, 16945192, 16945193, 16945194, 16945195, 16945196, 16945197, 16945198, 16945199, 16945200, 16945201, 16945202, 16945203, 16945204, 16945205, 16945206, 16945207, 16945208, 16945209, 16945210, 16945211, 16945212, 16945213, 16945214, 16945215, 16945216, 16945217, 16945218, 16945219, 16945220, 16945221, 16945222, 16945223, 16945224, 16945225, 16945226, 16945227, 16945228, 16945229, 16945230, 16945231, 16945232, 16945233, 16945234, 16945235, 16945236, 16945237, 16945238, 16945239, 16945240, 16945241, 16945242, 16945243, 16945244, 16945245, 16945246, 16945247, 16945248, 16945249, 16945250, 16945251, 16945252, 16945253, 16945254, 16945255, 16945256, 16945257, 16945258, 16945259, 16945260, 16945261, 16945262, 16945263, 16945264, 16945265, 16945266, 16945267, 16945268, 16945269, 16945270, 16945271, 16945272, 16945273, 16945274, 16945275, 16945276, 16945277, 16945278, 16945279, 16945280, 16945281, 16945282, 16945283, 16945284, 16945285, 16945287, 16945288, 16945289, 16945290, 16945291, 16945292, 16945293, 16945294, 16945295, 16945296, 16945297, 16945298, 16945299, 16945300, 16945301, 16945302, 16945303, 16945304, 16945305, 16945306, 16945307, 16945308, 16945309, 16945310, 16945311, 16945312, 16945313, 16945314, 16945315, 16945316, 16945317, 16945318, 16945319, 16945320, 16945321, 16945322, 16945323, 16945324, 16945325, 16945326, 16945327, 16945328, 16945329, 16945330, 16945331, 16945332, 16945333, 16945334, 16945335, 16945336, 16945337, 16945338, 16945339, 16945340, 16945341, 16945342, 16945343, 16945344, 16945345, 16945346, 16945347, 16945348, 16945349, 16945350, 16945351, 16945352, 16945353, 16945354, 16945355, 16945356, 16945357, 16945358, 16945359, 16945360, 16945361, 16945362, 16945363, 16945364, 16945365, 16945366, 16945367, 16945368, 16945369, 16945370, 16945371, 16945372, 16945373, 16945374, 16945375, 16945376, 16945377, 16945378, 16945379, 16945380, 16945381, 16945382, 16945383, 16945384, 16945385, 16945386, 16945387, 16945388, 16945389, 16945390, 16945391, 16945392, 16945393, 16945394, 16945395, 16945396, 16945397, 16945398, 16945399, 16945400, 16945401, 16945402, 16945403, 16945404, 16945405, 16945406, 16945407, 16945408, 16945409, 16945410, 16945411, 16945412, 16945413, 16945414, 16945415, 16945416, 16945417, 16945419, 16945420, 16945421, 16945422, 16945423, 16945424, 16945425, 16945426, 16945427, 16945428, 16945429, 16945430, 16945431, 16945432, 16945433, 16945434, 16945435, 16945436, 16945437, 16945438, 16945440, 16945441, 16945442, 16945443, 16945444, 16945445, 16945446, 16945447, 16945448, 16945449, 16945450, 16945451, 16945452, 16945453, 16945454, 16945455, 16945456, 16945457, 16945458, 16945459, 16945460, 16945461, 16945462, 16945463, 16945464, 16945465, 16945466, 16945467, 16945468, 16945469, 16945470, 16945471, 16945472, 16945473, 16945474, 16945475, 16945476, 16945477, 16945478, 16945479, 16945480, 16945481, 16945482, 16945483, 16945484, 16945485, 16945486, 16945487, 16945488, 16945489, 16945490, 16945491, 16945492, 16945493, 16945494, 16945495, 16945496, 16945497, 16945498, 16945499, 16945500, 16945501, 16945502, 16945503, 16945504, 16945505, 16945506, 16945507, 16945508, 16945509, 16945510, 16945511, 16945512, 16945513, 16945514, 16945515, 16945516, 16945517, 16945518, 16945519, 16945520, 16945521, 16945522, 16945523, 16945524, 16945525, 16945526, 16945527, 16945528, 16945529, 16945530, 16945531, 16945532, 16945533, 16945534, 16945535, 16945536, 16945537, 16945538, 16945539, 16945540, 16945541, 16945542, 16945543, 16945544, 16945545, 16945546, 16945547, 16945548, 16945549, 16945550, 16945551, 16945552, 16945553, 16945554, 16945555, 16945556, 16945557, 16945558, 16945559, 16945560, 16945561, 16945562, 16945563, 16945564, 16945565, 16945566, 16945567, 16945568, 16945569, 16945570, 16945571, 16945572, 16945573, 16945574, 16945575, 16945576, 16945577, 16945578, 16945579, 16945580, 16945581, 16945582, 16945583, 16945584, 16945585, 16945586, 16945587, 16945588, 16945589, 16945590, 16945591, 16945592, 16945593, 16945594, 16945595, 16945596, 16945597, 16945598, 16945599, 16945600, 16945601, 16945602, 16945603, 16945604, 16945605, 16945606, 16945607, 16945608, 16945609, 16945610, 16945611, 16945612, 16945613, 16945614, 16945615, 16945616, 16945617, 16945618, 16945619, 16945620, 16945621, 16945622, 16945623, 16945624, 16945625, 16945626, 16945627, 16945628, 16945629, 16945630, 16945631, 16945632, 16945633, 16945634, 16945635, 16945636, 16945637, 16945638, 16945639, 16945640, 16945641, 16945642, 16945643, 16945644, 16945645, 16945646, 16945647, 16945648, 16945649, 16945650, 16945651, 16945652, 16945653, 16945654, 16945655, 16945656, 16945657, 16945658, 16945659, 16945660, 16945661, 16945662, 16945663, 16945664, 16945665, 16945666, 16945667, 16945668, 16945669, 16945670, 16945671, 16945672, 16945673, 16945674, 16945675, 16945676, 16945677, 16945678, 16945679, 16945680, 16945681, 16945682, 16945683, 16945684, 16945685, 16945686, 16945687, 16945688, 16945689, 16945690, 16945691, 16945692, 16945693, 16945694, 16945695, 16945696, 16945697, 16945698, 16945699, 16945700, 16945701, 16945702, 16945703, 16945704, 16945705, 16945706, 16945707, 16945708, 16945709, 16945710, 16945711, 16945712, 16945713, 16945714, 16945715, 16945716, 16945717, 16945718, 16945719, 16945720, 16945721, 16945722, 16945723, 16945724, 16945725, 16945728, 16945729, 16945730, 16945731, 16945732, 16945733, 16945734, 16945735, 16945736, 16945737, 16945738, 16945739, 16945740, 16945741, 16945742, 16945743, 16945744, 16945745, 16945746, 16945747, 16945748, 16945749, 16945750, 16945751, 16945752, 16945753, 16945754, 16945755, 16945756, 16945757, 16945758, 16945759, 16945760, 16945761, 16945762, 16945763, 16945764, 16945765, 16945766, 16945767, 16945768, 16945769, 16945770, 16945771, 16945772, 16945773, 16945774, 16945775, 16945776, 16945777, 16945778, 16945779, 16945780, 16945781, 16945782, 16945783, 16945784, 16945785, 16945786, 16945787, 16945788, 16945789, 16945790, 16945791, 16945792, 16945793, 16945794, 16945795, 16945796, 16945797, 16945798, 16945799, 16945800, 16945801, 16945802, 16945803, 16945804, 16945805, 16945806, 16945807, 16945808, 16945809, 16945810, 16945811, 16945812, 16945813, 16945814, 16945815, 16945816, 16945817, 16945818, 16945819, 16945820, 16945821, 16945822, 16945823, 16945824, 16945825, 16945826, 16945827, 16945828, 16945829, 16945830, 16945831, 16945832, 16945833, 16945834, 16945835, 16945836, 16945837, 16945838, 16945839, 16945840, 16945841, 16945842, 16945843, 16945844, 16945845, 16945846, 16945847, 16945848, 16945849, 16945850, 16945851, 16945852, 16945853, 16945854, 16945855, 16945856, 16945857, 16945858, 16945859, 16945860, 16945861, 16945862, 16945863, 16945864, 16945865, 16945866, 16945867, 16945868, 16945869, 16945870, 16945871, 16945872, 16945873, 16945874, 16945875, 16945876, 16945877, 16945878, 16945879, 16945880, 16945881, 16945882, 16945883, 16945884, 16945885, 16945886, 16945887, 16945888, 16945889, 16945890, 16945891, 16945892, 16945893, 16945894, 16945895, 16945896, 16945897, 16945898, 16945899, 16945900, 16945901, 16945902, 16945903, 16945904, 16945905, 16945906, 16945907, 16945908, 16945909, 16945910, 16945911, 16945912, 16945913, 16945914, 16945915, 16945916, 16945917, 16945918, 16945919, 16945920, 16945921, 16945922, 16945923, 16945924, 16945925, 16945926, 16945927, 16945928, 16945929, 16945930, 16945931, 16945932, 16945933, 16945934, 16945935, 16945936, 16945937, 16945938, 16945939, 16945940, 16945941, 16945942, 16945943, 16945944, 16945945, 16945946, 16945947, 16945948, 16945949, 16945950, 16945951, 16945952, 16945953, 16945954, 16945955, 16945956, 16945957, 16945958, 16945959, 16945960, 16945961, 16945962, 16945963, 16945964, 16945965, 16945966, 16945967, 16945968, 16945969, 16945970, 16945971, 16945972, 16945973, 16945974, 16945975, 16945976, 16945977, 16945978, 16945979, 16945980, 16945981, 16945982, 16945983, 16945984, 16945985, 16945986, 16945987, 16945988, 16957658, 16957682, 16957689, 16957690, 16957696, 16957702, 16957703, 16957715, 16957716, 16957717, 16957723, 16957724, 16957731, 16957738, 16967478, 16972651]

for i in range (0,(len(lista)+1)):

    driver.get(f'https://etherscan.io/block/{lista[i]}')
    v1=True
    while v1:
        try: 
            time.sleep(0.3)                                                                           
            blocknumber_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[1]/div[2]/div/div[1]')
            v1=False
        except NoSuchElementException:

            break 
                
        blocknumber=0
    # v2=True    
    # while v2:    
    #     try:             
    #         time.sleep(0.3)                                                               
    #         blockstatus_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[2]/div[2]/span')
    #         v2=False
    #     except NoSuchElementException:
    #         break
    # v3=True    
    # while v3:    
    #     try:
    #         time.sleep(0.3)
    #         blockdateutc_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[3]/div[2]')
    #         v3=False
    #     except NoSuchElementException:
    #         break
    
    # v4=True
    # while v4:
    #     try:
    #         time.sleep(0.3)
    #         blockslot_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_divhSlotEpoch"]/div[2]/a[1]')
    #         v4=False
    #     except NoSuchElementException:   

    #         break
    # v5=True
    # while v5:   
         
    #     try:
    #         time.sleep(0.3)
    #         blockepock_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_divhSlotEpoch"]/div[2]/a[2]')
    #         v5=False
    #     except NoSuchElementException:
    #         break
    v6=True    
    while v6:    
        try:
            time.sleep(0.3)
            blocktrxnbr_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_div_tx_fieldvalue"]/a[1]')
            v6=False
        except NoSuchElementException:
            break

    # v7=True    
    # while v7:    
    #         try:
    #             time.sleep(0.3)
    #             blockreceipient_address_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[6]/div[2]/a[1]')
    #             V7=False
    #         except NoSuchElementException:
    #             break
    v8=True
    while v8:        
            try:
                time.sleep(0.3)
                block_reward_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[7]/div[2]')
                v8=False
            except NoSuchElementException:
                break
    v9=True        
    while v9:        
            try:
                time.sleep(0.3)
                trnx_fees_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[7]/div[2]/span[2]')
                v9=False
            except NoSuchElementException:
                break
    v10=True        
    while v10:        
                try:
                      time.sleep(0.3)
                      burnt_fees_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[7]/div[2]/span[3]')
                      v10=False
                except NoSuchElementException:
                    break
    # v11=True            
    # while v11:            
    #         try:
    #              time.sleep(0.3)
    #              blockdifficulty_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[8]/div[2]')
    #              v11=False
    #         except NoSuchElementException:
    #             break
    v12=True        
    while v12:        
            try:
               time.sleep(0.3)
               blocksize_element  = driver.find_element(By.XPATH, ' //*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[9]/div[2]')
               v12=False
            except NoSuchElementException:

                break
    v13=True        
    while v13:        
            try:
                time.sleep(0.3)
                block_gasused_element  = driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[10]/div[2]')
                v13=False
            except NoSuchElementException:
                break
    # v14=True        
    # while v14:        
    #         try:
                
    #              time.sleep(0.3)
    #              block_gasused_porcentage_element  = driver.find_element(By.XPATH, ' //*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[10]/div[2]/span')
    #              v14=False
    #         except NoSuchElementException:
    #             break
    # v15=True        
    # while v15:
    #             try:
    #                 time.sleep(0.3)
    #                 block_gaslimit_element  = driver.find_element(By.XPATH, ' //*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[11]/div[2]')
    #                 v15=False
    #             except NoSuchElementException:
    #                 break
    v16=True            
    while v16:
            try:
                 time.sleep(0.3)
                 base_fee_per_gas_ETH_element  = driver.find_element(By.XPATH, ' //*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[12]/div[2]')
                 v16=False
            except NoSuchElementException:
                break
    v17=True
    while v17:
            try:
                 time.sleep(0.3)
                 base_fee_per_gas_GWEI_element  = driver.find_element(By.XPATH, ' //*[@id="ContentPlaceHolder1_maintable"]/div[1]/div[12]/div[2]/span')
                 v17=False
            except NoSuchElementException:
                break
    
    try:
                
        blocknumber=blocknumber_element.text.strip()
    except:        
         blocknumber=0
    # try:
                
    #     blockstatus=blockstatus_element.text.strip()
    # except:
    #      blockstatus=0
    # try:
    #      blockdateutc=blockdateutc_element.text.strip()
    # except:
              
    #      blockdateutc=0   
    # try:
                
    #     blockslot=blockslot_element.text.strip()
    # except:
    #      blockslot=0
    # try:
                
    #     blockepock=blockepock_element.text.strip()
    # except: 
    #      blockepock=0   
    try:
                
        blocktrxnbr=int(blocktrxnbr_element.text.strip().split("transactions")[0].strip())
    except: 
         blocktrxnbr=0 
    # try:
                
    #     blockreceipient_address=blockreceipient_address_element.text.strip()
    # except: 
    #      blockreceipient_address=0 
    try:
                
        block_reward=block_reward_element.text.strip().split("ETH")[0].strip()
    except: 
         block_reward=0   
    try:
                
        trnx_fees=trnx_fees_element.text.strip()
    except: 
         trnx_fees=0 
    try:
                
        burnt_fees=burnt_fees_element.text.strip()[1:]
    except: 
         burnt_fees=0 
    # try:
                
    #     blockdifficulty=blockdifficulty_element.text.strip()
    # except: 
    #      blockdifficulty=0 
    try:
                
        blocksize=blocksize_element.text.strip().split("bytes")[0].strip()
    except: 
         blocksize=0        
    try:
                
        block_gasused=block_gasused_element.text.strip().split("(")[0].strip()
    except: 
         block_gasused=0
    # try:
                
    #     block_gasused_porcentage=block_gasused_porcentage_element.text.strip()[1:-2]
    # except: 
    #      block_gasused_porcentage=0
    # try:
                
    #     block_gaslimit=block_gaslimit_element.text.strip()
    # except: 
    #      block_gaslimit=0
    try:
                
        base_fee_per_gas_ETH=base_fee_per_gas_ETH_element.text.strip().split("ETH")[0].strip()
    except: 
         base_fee_per_gas_ETH=0     
    try:
                
        base_fee_per_gas_GWEI=base_fee_per_gas_GWEI_element.text.strip().split("(")[1].strip().split("Gwei")[0].strip()
    except: 
         base_fee_per_gas_GWEI=0 
    # try:
    #     blockt_timestamp=blockdateutc_element.text.strip().split("(")[1].strip().split("+UTC")[0].strip()
    #     date_string = blockt_timestamp[0:len(blockt_timestamp)-2].strip()
    #     date_format = "%b-%d-%Y %H:%M:%S"

    #     # Convert the date string to a datetime object
    #     date_object = datetime.strptime(date_string, date_format)

    #     # Convert the datetime object to a Unix timestamp
    #     blockt_timestamp = int(date_object.timestamp())

    # except:
    #     blockt_timestamp=0                                                           

    
    print("___________________________________newblock :",blocknumber ,"___________________________________________ ")
    print("blocknumber:",blocknumber)
    # print("blockstatus:",blockstatus)
    # print("blockt_timestamp:",blockt_timestamp)
    # print("blockslot:",blockslot)
    # print("blockepock:",blockepock)
    print("transactions_nbr:",blocktrxnbr)
    print("blockreward_ETH;",block_reward)
    print("trnx_fees;",trnx_fees)
    print("burnet_fees;",burnt_fees)
    # print("block_total_difficulty;",blockdifficulty)
    print("blocksize_bytes;",blocksize)
    print("blockgasused:",block_gasused)
    # print("blockgasused_poucentage:",block_gasused_porcentage)
    # print("blockgaslimit:",block_gaslimit)
    print("block_base_fee_per_gas_ETH:",base_fee_per_gas_ETH)
    print("block_base_fee_per_gas_Gwei:",base_fee_per_gas_GWEI)
    # print("block_burnet_fees_ETH:",base_fee_per_gas_ETH)
    print("-----------------------------------------------------------------")
    timestampp = time.time()
    formatted_time3 = datetime.fromtimestamp(timestampp).strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time3)
    my_dict={
            "timeStamp":formatted_time3,
            "blocknumber:":blocknumber,
            # "blockstatus:":blockstatus,
            # "blockt_timestamp:":blockt_timestamp,
            # "blockslot:":blockslot,
            # "blockepock:":blockepock,
            "transactions_nbr:":blocktrxnbr,
            "blockreward_ETH;":block_reward,
            "trnx_fees;":trnx_fees,
            "burnet_fees;":burnt_fees,
            # "block_total_difficulty;":blockdifficulty,
            "blocksize_bytes;":blocksize,
            "blockgasused:":block_gasused,
            # "blockgasused_poucentage:":block_gasused_porcentage,
            # "blockgaslimit:":block_gaslimit,
            # "block_base_fee_per_gas_Gwei:":base_fee_per_gas_GWEI,
            "block_base_fee_per_gas_ETH:":base_fee_per_gas_ETH,
            # "block_burnet_fees_ETH:":base_fee_per_gas_ETH,   
        }
    
    df1=pd.DataFrame(my_dict,index=[0])
    generalinf_df= pd.concat([generalinf_df,df1])
    print(generalinf_df)
    generalinf_df.to_csv("Bloc4.csv", index=False)
    # time.sleep(7)
