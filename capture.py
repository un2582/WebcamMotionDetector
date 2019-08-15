import cv2, time, pandas
from datetime import datetime
times = [];
video = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW
first_frame = None
status_list = [None,None]
df = pandas.DataFrame()

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame, 110, 255, cv2.THRESH_BINARY)[1] #turn pixels within threshold of 110 to 255
    thresh_delta = cv2.dilate(thresh_delta,None, iterations = 2) #helps dilate the area
    #get a copy of thresh delta, find its contours and put it into a variable
    (cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #now for each contour in the cnts variable, do the following
    for contour in cnts:
        #if the area of the contour is less than a certain amount don't count it
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1
        #if the white contour area is huge, bind its coordinates into a rectangle
        (x, y, w, h) = cv2.boundingRect(contour)
        #draw the rectangle in green color
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)

    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    #cv2.imshow("capturing", gray)
    #cv2.imshow("frame difference",delta_frame)
    cv2.imshow("Threshhold frames", thresh_delta)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        status = 1
        times.append(datetime.now())
        if len(times) % 2 != 0:                 # check if "times" list is odd
            del times[0] #this ensures that I my list indices never give an error since it requires even # of entries
        break

print(status_list)
print(times)
for i in range(0, len(times), 2): #0 1 2 3 4 5
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("times1.csv")
video.release()
cv2.destroyAllWindows()
