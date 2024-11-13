import numpy as np
import soundfile as sf
import sounddevice as sd

# 샘플레이트 및 주기
Fs = 44100
Ts = 1 / Fs

# Op-amp stage
C1 = 10e-9
R1 = Ts / (2 * C1)

C2 = 47e-9
R2 = Ts / (2 * C2)

R3 = 10e3
R4 = 1e6
R5 = 4.7e3
R8 = 1e6

# "Distortion" 포지션 설정 (0에서 1 사이)
pot = 0.75
k = 8
R6 = (np.exp(-k * pot) - np.exp(-k)) / (1 - np.exp(-k)) * 1000000
Rn = R5 + R6

# 전달 함수의 G 값
Ga = 1 + (R3 / R1)
Gb = 1 + (R2 / Rn)
Gh = 1 + (R4 / (Gb * Rn))
Gx = (1 / R8) + (1 / (Ga * R1))

# 전달 함수의 계수
b0 = Gh / (Ga * R1 * Gx)
b1 = ((R3 / (Ga * R1)) - 1) * (Gh / Gx)
b2 = (-R2 * R4) / (Gb * Rn)

# 입력 신호: 오디오 파일
input_signal, Fs = sf.read('testaudio.wav')
N = len(input_signal)

# 출력 신호 설정
y = np.zeros(N)

# 초기 상태값 설정
x1 = 0
x2 = 0

# Op-amp 단계의 샘플별 처리
for n in range(N):
    Vin = input_signal[n]
    Vout = b0 * Vin + b1 * x1 + b2 * x2

    Vx = (1 / Gh) * Vout + ((R2 * R4) / (Gb * Gh * Rn)) * x2
    VR1 = (Vin - Vx + (R3 * x1)) / Ga

    VRn = (Vx - (R2 * x2)) / Gb
    VR2 = (R2 / Rn) * VRn + (R2 * x2)

    x1 = (2 / R1) * VR1 - x1
    x2 = (2 / R2) * VR2 - x2

    y[n] = Vout

# Clipping stage
Is = 100e-9
Vt = 0.026
eta = 2

Rb = 10000
Ca = 1e-9
Ra = Ts / (2 * Ca)

outputpot = 0.76
Re = 10000 * (1 - np.log10(1 + 9 * (1 - outputpot)))
Rd = 10000 - Re

Gg = (1 / Rd) + (1 / Re)

Vd = 0
Vout2 = Vd / (Rd * Gg)
TOL = 1e-10
xa = 0

z = np.zeros(N)

for n in range(N):
    Vin2 = y[n]

    fVd = 2 * Is * np.sinh(Vd / (eta * Vt)) + (Vd / Rb) + (-Vin2 / Rb) + (Vd / Ra) - xa + (Vd / Rd) + (-Vout2 / Rd)
    count = 0

    while abs(fVd) > TOL and count < 10:
        der = ((2 * Is / (eta * Vt)) * np.cosh(Vd / (eta * Vt))) + (1 / Ra) + (1 / Rb) + (1 / Rd)
        Vd -= fVd / der
        fVd = 2 * Is * np.sinh(Vd / (eta * Vt)) + (Vd / Rb) + (-Vin2 / Rb) + (Vd / Ra) - xa + (Vd / Rd) + (-Vout2 / Rd)
        count += 1

    Vout2 = Vd / (Rd * Gg)
    xa = ((2 / Ra) * Vd) - xa

    z[n] = Vout2

# 오디오 파일로 저장
sf.write('input.wav', input_signal, Fs)
sf.write('output.wav', z, Fs)

# 오디오 재생
print("Input signal playing...")
sd.play(input_signal, Fs)
sd.wait()  # 재생이 끝날 때까지 기다림

print("Processed output signal playing...")
sd.play(z, Fs)
sd.wait()  # 재생이 끝날 때까지 기다림
