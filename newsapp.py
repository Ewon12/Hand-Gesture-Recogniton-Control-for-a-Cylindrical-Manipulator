import cv2
import numpy as np
import pickle
import time
import serial
import PySimpleGUI as sg

sg.theme('DarkBrown4')

intro = [
   
    [sg.Push(),sg.Image(filename='hgrc.png',size = (720,530)),sg.Push()], [sg.VerticalSeparator(pad=(0, 10))],
    [sg.Push(),sg.Button('RUN', disabled=False ,font=("LCDMono2 Bold",30),button_color=('black','pink')),sg.Push()],
    [sg.Push(),sg.Exit(font = ('Times New Roman',13))]

]

intro_windows =sg.Window('MEXE 004',intro)

def main_layout():
    sg.theme('DarkBrown4')
    Main_layout = [
        [sg.Push(),sg.Button('HELP', disabled=False ,font=("Arial Bold",10),button_color=('black','white')),
        sg.Button('ABOUT', disabled=False ,font=("Arial Bold",10),button_color=('black','white'))],
        [sg.Push(),sg.Text('HAND GESTURE RECOGNITION CONTROL FOR A CYLINDRICAL MANIPULATOR',
        font =("Dungeon Bold",20),text_color='orange'),sg.Push()],[sg.VerticalSeparator(pad=(0, 10))],
        
            [sg.Push(),sg.Frame('Fill out the following: ',[
                [sg.Text('Filename for training data: ', font = ('Times New Roman',15)),sg.InputText('default',key='default',size=(15,10))],
                [sg.Text('SERIAL PORT: ', font = ('Times New Roman',15)),sg.InputText('COM5',key='SERIAL',size=(15,10))],
                [sg.Text('CAMERA: ', font = ('Times New Roman',15)),sg.InputText('0',key='CAM',size=(15,10))],
                [sg.Text('HEIGHT for FRAME: ', font = ('Times New Roman',15)),sg.InputText('640',key='height',size=(15,10))],
                [sg.Text('WIDTH for FRAME: ', font = ('Times New Roman',15)),sg.InputText('920',key='width',size=(15,10))]]),sg.Push(),
            sg.Frame('GESTURE AVAILABLE:',[
                [sg.Text('UP ', font = ('Times New Roman',15)),
                sg.Text('DOWN ', font = ('Times New Roman',15))],
                [sg.Text('RIGHT ', font = ('Times New Roman',15)),
                sg.Text('LEFT ', font = ('Times New Roman',15))],
                [sg.Text('FORWARD ', font = ('Times New Roman',15)),
                sg.Text('BACKWARD ', font = ('Times New Roman',15))],
                [sg.Text('GRIP ', font = ('Times New Roman',15)),
                sg.Text('UNGRIP ', font = ('Times New Roman',15))],
                [sg.Text('RESET ', font = ('Times New Roman',15)),
                sg.Text('GO AUTO ', font = ('Times New Roman',15))]]),sg.Push()],
            [sg.VerticalSeparator(pad=(0, 10))],
            [sg.Push(),sg.Button('RECOGNIZE', disabled=False ,font=("LCDMono2 Bold",20),button_color=('orange','purple')),sg.Push(),
            sg.Button('UPLOAD', disabled=False ,font=("LCDMono2 Bold",20),button_color=('blue','green')),sg.Push(),
            sg.Button('TRAIN', disabled=False ,font=("LCDMono2 Bold",20),button_color=('yellow','brown')),sg.Push()],
            [sg.VerticalSeparator(pad=(0, 10))],
            [sg.Push(),sg.Exit(font = ('Times New Roman',13))]

    ]
    
    window = sg.Window('HAND GESTURE RECOGNITION CONTROL', Main_layout,resizable =False)
    while True:
            event, values =window.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
    window.close()

while True:
        event, values =intro_windows.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        elif event == 'RUN':
            intro_windows.hide()
            main_layout()
            intro_windows.un_hide()

intro_windows.close() 