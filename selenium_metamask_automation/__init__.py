from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
import os
import urllib.request
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC, ui
from selenium.webdriver.support.wait import WebDriverWait

from config import global_config

EXTENSION_PATH = os.path.abspath(
    r"..") + '/selenium_metamask_auto_testing/selenium_metamask_automation/extension_metamask.crx'

EXTENSION_ID = 'nkbihfbeogaeaoehlefnkodbefgpgknn'


def downloadMetamaskExtension():
    print('Setting up metamask extension please wait...')

    url = 'https://xord-testing.s3.amazonaws.com/selenium/10.0.2_0.crx'
    urllib.request.urlretrieve(url, os.getcwd() + '/extension_metamask.crx')


def launchSeleniumWebdriver(driverPath,browser_type):
    print('path', EXTENSION_PATH)
    if browser_type == 'chrome':
      options = Options()
    elif browser_type == 'edge':
      options = webdriver.EdgeOptions()
    options.add_argument('blink-settings=imagesEnabled=false')
    options.add_argument('--disable-images')
    # 屏蔽webdriver特征
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # -------------------------------------------------------
    options.add_extension(EXTENSION_PATH)
    global driver
    driver = webdriver.Chrome(options=options,
                              executable_path=driverPath)
    # time.sleep(5)
    print("Extension has been loaded")
    return driver




def checkHandles():
    handles_value = driver.window_handles
    if len(handles_value) > 1:
        driver.switch_to.window(driver.window_handles[1])
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        checkHandles()


def metamaskSetup(recovery_phrase, password):
    cur_handles = driver.window_handles
    search_window = driver.current_window_handle
    wait_new_windows(cur_handles)
    time.sleep(1)
    is_visible('//button[text()="开始使用"]')
    driver.find_element("xpath", '//button[text()="开始使用"]').click()
    driver.find_element("xpath", '//button[text()="导入钱包"]').click()
    driver.find_element("xpath", '//button[text()="我同意"]').click()
    is_visible('//input')
    inputs = driver.find_elements("xpath", '//input')
    inputs[0].send_keys(recovery_phrase)
    inputs[1].send_keys(password)
    inputs[2].send_keys(password)
    driver.find_element(By.CSS_SELECTOR, '.first-time-flow__terms').click()
    driver.find_element("xpath", '//button[text()="导入"]').click()

    # time.sleep(8)

    driver.find_element("xpath", '//button[text()="全部完成"]').click()
    # time.sleep(2)

    # # closing the message popup after all done metamask screen
    # driver.find_element("xpath", '//*[@id="popover-content"]/div/div/section/header/div/button').click()
    # time.sleep(2)
    print("Wallet has been imported successfully")
    # time.sleep(1)
    driver.close()
    all_handles = driver.window_handles
    for handle in all_handles:
        if handle == search_window:
            driver.switch_to.window(handle)  # 跳转到百度首页窗口
            driver.refresh()


def changeMetamaskNetwork(networkName):
    # opening network
    print("Changing network")
    driver.execute_script("window.open();")
    driver.switch_to.window(driver.window_handles[1])
    driver.get('chrome-extension://{}/home.html'.format(EXTENSION_ID))
    driver.find_element(
        "xpath", '//*[@id="popover-content"]/div/div/section/header/div/button').click()
    # 打开网络下拉框
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[1]/div/div[2]/div[1]/div/span').click()
    # 跳转开启测试网设置
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[2]/div/div[1]/div[3]/span/a').click()
    # 显示测试网
    driver.find_element("xpath",
                        '//*[@id="app-content"]/div/div[3]/div/div[2]/div[2]/div[2]/div[7]/div[2]/div/div/div[1]/div[2]/div').click()
    # 滑到最上方
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    # 打开网络下拉框
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[1]/div/div[2]/div[1]/div/span').click()
    print("opening network dropdown")
    time.sleep(2)
    # 以太坊 Ethereum 主网络
    # Ropsten 测试网络
    # Kovan 测试网络
    # Rinkeby 测试网络
    # Goerli 测试网络
    all_li = driver.find_elements_by_tag_name('li')

    for li in all_li:
        text = li.text
        if text == networkName:
            li.click()
            print(text, "is selected")
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            return
    print("Please provide a valid network name")
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(3)


def addAndChangeNetwork(cur_handles):
    wait_new_windows(cur_handles)
    print("添加并切换网络开始")
    change_window_handle()
    is_visible("//button[text()='批准']")
    driver.find_element("xpath", "//button[text()='批准']").click()
    is_visible("//button[text()='切换网络']")
    driver.find_element("xpath", "//button[text()='切换网络']").click()


def changeNetworkByChainList(network_name):
    """
    通过Chainlist.org切换指定网络

    :Args:
        - network_name: string 完整的网络名.

    :Usage:
        auto.changeNetworkByChainList('Binance Smart Chain Mainnet')
    """
    time.sleep(5)
    print("切换指定网络开始")
    driver.execute_script("window.open();")
    driver.switch_to.window(driver.window_handles[1])
    driver.get('https://chainlist.org/')
    driver.find_element("xpath", "//h5[text()='Connect Wallet']").click()
    # connect chainlist
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[2])
    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(3)
    driver.find_element("xpath", '//button[text()="下一步"]').click()
    driver.find_element("xpath", '//button[text()="连接"]').click()
    driver.close()
    driver.switch_to.window(driver.window_handles[1])
    # search Network
    driver.find_element("xpath", "//span[text()='Testnets']").click()
    time.sleep(1)
    inputs = driver.find_element("xpath", '//input')
    inputs[0].send_keys(network_name)
    time.sleep(1)
    driver.find_element("xpath", "//span[text()='Add to Metamask']").click()
    # change Network
    time.sleep(3)
    driver.execute_script("window.open();")
    driver.switch_to.window(driver.window_handles[2])
    driver.get('chrome-extension://{}/home.html'.format(EXTENSION_ID))
    driver.find_element("xpath", "//button[text()='批准']").click()
    driver.find_element("xpath", "//button[text()='切换网络']").click()
    time.sleep(3)
    driver.close()
    driver.switch_to.window(driver.window_handles[1])
    driver.close()
    driver.switch_to.window(driver.window_handles[0])


def connectToWebsite(cur_handles):
    time.sleep(2)
    change_window_handle()
    driver.find_element("xpath", '//button[text()="下一步"]').click()
    driver.find_element("xpath", '//button[text()="连接"]').click()
    is_visible('//button[text()="签名"]')
    driver.find_element("xpath", '//button[text()="签名"]').click()
    print('Site connected to metamask')
    driver.switch_to.window(driver.window_handles[0])


def confirmApprovalFromMetamask():
    time.sleep(5)
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    driver.find_element("xpath", '//button[text()="确认"]').click()
    print("Approval transaction confirmed")

    driver.close()
    # switch back
    driver.switch_to.window(driver.window_handles[0])


def rejectApprovalFromMetamask():
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    # time.sleep(10)
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    # time.sleep(10)
    # confirm approval from metamask
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[3]/div/div[4]/footer/button[1]').click()
    time.sleep(8)
    print("Approval transaction rejected")

    # switch to dafi
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(3)
    print("Reject approval from metamask")


def confirmTransactionFromMetamask():
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    time.sleep(10)
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(10)

    # # confirm transaction from metamask
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[3]/div/div[3]/div[3]/footer/button[2]').click()
    time.sleep(13)
    print("Transaction confirmed")

    # switch to dafi
    driver.switch_to.window(driver.window_handles[0])

    time.sleep(3)


def rejectTransactionFromMetamask():
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    time.sleep(5)
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(5)
    # confirm approval from metamask
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[3]/div/div[3]/div[3]/footer/button[1]').click()
    time.sleep(2)
    print("Transaction rejected")

    # switch to web window
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(3)


def addToken(tokenAddress):
    # opening network
    print("Adding Token")
    driver.switch_to.window(driver.window_handles[1])
    driver.get('chrome-extension://{}/home.html'.format(EXTENSION_ID))
    print("closing popup")
    time.sleep(5)
    driver.find_element(
        "xpath", '//*[@id="popover-content"]/div/div/section/header/div/button').click()

    # driver.find_element("xpath", '//*[@id="app-content"]/div/div[1]/div/div[2]/div[1]/div/span').click()
    # time.sleep(2)

    print("clicking add token button")
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[4]/div/div/div/div[3]/div/div[3]/button').click()
    time.sleep(2)
    # adding address
    driver.find_element(By.ID, "custom-address").send_keys(tokenAddress)
    time.sleep(10)
    # clicking add
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[4]/div/div[2]/div[2]/footer/button[2]').click()
    time.sleep(2)
    # add tokens
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[4]/div/div[3]/footer/button[2]').click()
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(3)


def signConfirm():
    time.sleep(5)
    checkHandles()
    time.sleep(1)

    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(5)
    while True:
        try:
            element = driver.find_element(
                "xpath", '//*[@id="app-content"]/div/div[2]/div/div[3]/div[1]')
        except NoSuchElementException:
            time.sleep(1)
            print('签名了，但没有完全签名')
            break
        else:
            driver.find_element(
                "xpath", '//*[@id="app-content"]/div/div[2]/div/div[3]/div[1]').click()
            driver.find_element("xpath", '//button[text()="签名"]').click()
            time.sleep(1)
            print('签名完成')
            break
    driver.close()
    driver.switch_to.window(driver.window_handles[0])


def signReject():
    print("sign")
    time.sleep(3)

    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])

    driver.get('chrome-extension://{}/popup.html'.format(EXTENSION_ID))
    time.sleep(5)
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(3)
    driver.find_element(
        "xpath", '//*[@id="app-content"]/div/div[3]/div/div[3]/button[1]').click()
    time.sleep(1)
    # driver.find_element("xpath", '//*[@id="app-content"]/div/div[3]/div/div[2]/div[2]/div[2]/footer/button[2]').click()
    # time.sleep(3)
    print('Sign rejected')
    print(driver.window_handles)
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(3)


# 一直等待某元素可见，默认超时10秒
def is_visible(locator, timeout=10):
    try:
        ui.WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.XPATH, locator)))
        return True
    except TimeoutException:
        return False


def wait_new_windows(cur_handles):
    # Ec条件 -- 等待新窗口出现
    WebDriverWait(driver, 20).until(
        EC.new_window_is_opened(cur_handles))  # 更稳定的窗口等待
    wins = driver.window_handles
    driver.switch_to.window(wins[-1])


def change_window_handle():
    wins = driver.window_handles
    driver.switch_to.window(wins[-1])
