import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

options = Options()
options.binary_location = "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
options.add_argument("--no-first-run")
options.add_argument("--no-default-browser-check")
options.add_argument("--user-data-dir=/tmp/selenium-profile")
# options.add_argument("--headless=new") # comment this out to see selenium window opened

service = Service("chromedriver-mac-arm64/chromedriver")
driver = webdriver.Chrome(service=service, options=options)

driver.get('http://www.google.com/')
time.sleep(2)

search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5)
driver.quit()
