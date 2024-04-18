import cv2, os, sys
from numpy import sin, cos, matrix, dot, array, zeros, float32
from math import radians
from PIL import Image

def wczytaj_obiekty(zoom = 1):
    fnames = os.listdir('obiekty/')
    paths = ['obiekty/' + x for x in fnames]
    obiekty = []
    for p in paths:
        with open(p, 'r') as f:
            text = f.read()
            f.close()
        numery = [float(x) for x in text.split()]
        if len(numery) != 6:
            print(f'Błędny format pliku {p}')
            exit()
        zoomed = tuple(zoom * x for x in numery)
        x0, y0, z0,  xlen, ylen, h = zoomed
        x1 = x0 + xlen
        y2 = y0 + ylen
        z4 = z0 + h
        punkty = [[x0, y0, z0], [x1, y0, z0], [x1, y2, z0], [x0, y2, z0],
                  [x0, y0, z4], [x1, y0, z4], [x1, y2, z4], [x0, y2, z4]]
        obiekty.append(punkty)
    return obiekty if len(obiekty) != 0 else print('katalog obiektów jest pusty'); exit()


def translacja(obiekty, xyzk):
    xk, yk, zk = xyzk
    for obiekt in obiekty:
        for punkt in obiekt:
            punkt[0] -= xk
            punkt[1] -= yk
            punkt[2] -= zk 
            
            
def rotacja(obiekty, obrot_kamery):
    rx, ry, rz = obrot_kamery
    for o in obiekty:
        for p in o:
            A = matrix([[p[0]], [p[1]], [p[2]]])
            po_x = dot(Rx(rx), A)
            po_y = dot(Ry(ry), po_x)
            po_z = dot(Rz(rz), po_y)
            p[0] = po_z[0, 0].item()
            p[1] = po_z[1, 0].item()
            p[2] = po_z[2, 0].item()

def Rx(a):
    return matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])

def Ry(a):
    return matrix([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])

def Rz(a):
    return matrix([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])


def projekcja(obiekty):
    m = matrix([[1, 0, 0], [0, 1 ,0], [0, 0, 0]])
    v = []
    for o in obiekty:
        obiekt_zrzutowany = []
        for p in o:
            A = matrix([[p[0]], [p[1]], [p[2]]])
            x = dot(m, A)
            obiekt_zrzutowany.append([x[0,0].item(), x[1,0].item(), x[2,0].item()])
        v.append(obiekt_zrzutowany)
    return v

# def projekcja(obiekty):
#     z_odl = 20
#     v = []
#     for o in obiekty:
#         obiekt_zrzutowany = []
#         for p in o:
#             print(z_odl)
#         v.append(obiekt_zrzutowany)
#     return v


def wylicz_wspolrzedne(wymiary, punkt):
    w, h = wymiary
    x, y = punkt
    x_centrum = int(w/2)
    y_centrum = int(h/2)
    x_wynik = int(x + x_centrum)
    y_wynik = int(y - y_centrum)
    return x_wynik, -y_wynik


def rysuj(obiekty, polaczenia):
    wielkosc_tla = (640,480)
    tlo = array(Image.new('RGB', wielkosc_tla))
    cv2.namedWindow('kamera', cv2.WINDOW_NORMAL)
    for o in obiekty:
        punkty = []
        for p in o:
            punkt = wylicz_wspolrzedne(wielkosc_tla, (p[0], p[1]))
            #print(type(punkt), type(punkt[0]), type(punkt[1]), punkt[0], punkt[1])
            cv2.circle(tlo, punkt, radius=0, color=(255,255,255), thickness=3)
            punkty.append(punkt)
        for i in range(len(polaczenia)):
            for j in polaczenia[i]:
                cv2.line(tlo, punkty[i], punkty[j], color=(255, 255, 255), thickness=1)
    cv2.imshow('kamera', tlo)
    while True:
        o = cv2.waitKey(0)
        if o == 32: 
            break
    cv2.destroyAllWindows()

        
if __name__ == "__main__":
    polaczenia = [[1, 3, 4], [0, 2, 5], [1, 3, 6], [0, 2, 7], [0, 5, 7], [1, 4, 6], [2, 5, 7], [3, 4, 6]]
    argumenty = sys.argv
    narg = len(argumenty)
    if narg != 5 and narg != 8:
        print('Argumenty uruchomienia odpowiadają za rozmieszczenie kamery, \n \
              w kolejności: x, y, z [rotacja po x, y, z]\n \
              normalnie osie układu kamery są takie same jak układu sceny')
        exit()
    xyz_kamera = [float(x) for x in argumenty[1:4]]
    if narg == 8:
        obrot_kamera = [radians(float(x)) for x in argumenty[4:7]]
    else: obrot_kamera = [0,0,0]
    zoom = float(argumenty[-1])
    ob = wczytaj_obiekty(zoom)
    translacja(ob, xyz_kamera)
    rotacja(ob, obrot_kamera)
    osie = [[[10,10,10]]]
    rotacja(osie, obrot_kamera)
    v = projekcja(ob)
    rysuj(v, polaczenia)