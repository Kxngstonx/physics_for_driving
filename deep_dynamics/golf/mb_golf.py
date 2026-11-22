"""CommonRoad 다물체 모델 vehicle_dynamics_mb 의 복사본 + 앞뒤 축별 타이어 계수 (Golf 시뮬레이터용).

원본: commonroad-vehicle-models 3.0.2, vehiclemodels/vehicle_dynamics_mb.py (TUM Cyber-Physical Systems Group,
BSD 라이선스 — 같은 폴더의 LICENSE_commonroad_vehicle_models.txt). 원본 패키지는 고치지 않았다.

바꾼 것 (아래 네 가지뿐. 그 밖의 줄은 원본과 같음):
  - 타이어 계수: 앞바퀴(LF, RF)는 p.tire_f, 뒷바퀴(LR, RR)는 p.tire_r (원본은 넷 다 같은 p.tire).
    이유: 원본 타이어는 횡강성이 하중에 정비례(K_y = p_ky1·F_z)라 앞뒤 축 강성 비가 하중 비와 같아져 언더스티어가 거의 0.
    실제 Golf 는 20~35 m/s 에서 조향 대비 회전 반응이 운동학 값의 0.44~0.48 배 (강한 언더스티어) [확인 11/18].
    축별 횡강성을 데이터 추정값(deep_dynamics/golf/specs.json 의 tire_stiffness_f/r, 11/21)에 맞추려고 나눔
  - [11/19] ABS: 바퀴별 미끄럼률(이 식 안의 s_lf..s_rr)이 p.abs_slip = (시작, 끝) 사이면 그 바퀴의 브레이크 토크를 1 → 0 으로 줄임.
    p.abs_slip 이 None 이면 원본과 같음. 이유: 원본은 브레이크 토크를 좌우 같게 나누고 잠김을 막지 않아, 코너 제동에서 하중이
    빠진 안쪽 뒷바퀴가 잠기고 차가 돈다(적분 간격과 무관). Golf 에는 ABS 가 있다(데이터 제동 시간의 1%에서 작동) [확인 11/18]
  - [11/21] 타이어 좌우 거울상 (선택, p.tire_mirror). 원본 tire_model.formula_lateral 은 옆 힘 밀림 항을
    sign(캠버)·(p_hy1, Fz·p_vy1) 로 둬서, 직진 중 차체가 0.1° 만 기울어도 바퀴마다 하중의 3.7% 옆 힘이 켜졌다 꺼졌다 한다
    (네 바퀴 부호가 같아지면 차가 옆으로 흐름) [확인 11/21]. p.tire_mirror=True 이면 표준 PAC2002 식(부호 함수 없음,
    _lateral_std)을 쓰고, 측정 쪽(왼쪽)과 반대인 오른쪽 바퀴는 타이어를 거울상으로 쓴다: Fy_R(κ, α, γ) = −Fy_L(κ, −α, −γ)
    (MBSim magic_formula_62 와 같은 규칙. Project Chrono ChPac02Tire·shift-dynamics pac2002 는 같은 효과를 계수 부호로 냄).
    [11/22 채택] 시뮬레이터 기본은 거울상 (sim.golf_params tire_mirror=True). False 면 원본과 같음 (비교 재현용:
    experiments/golf/compare_tire_mirror.py). 앞뒤 방향 힘은 그대로 (거울상 대상 아님)
  - [11/22] 앞 차동장치 LSD (선택, p.lsd_mode, 바퀴 회전 식 바로 위 설명). 원본은 앞 좌우에 구동 토크를 항상 반씩 (열린 차동장치).
    실제 차(Golf GTI Performance)는 전자식 LSD(VAQ)가 있고 데이터의 LSDengage_percent 가 가속 코너의 42% 에서 0 이 아님
"""
import math

from vehiclemodels.utils.steering_constraints import steering_constraints
from vehiclemodels.utils.acceleration_constraints import acceleration_constraints
from vehiclemodels.utils.vehicle_dynamics_ks_cog import vehicle_dynamics_ks_cog
import vehiclemodels.utils.tire_model as tireModel

__author__ = "Matthias Althoff"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2020a"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


def _lateral_std(alpha, gamma, F_z, p):
    """표준 PAC2002 순수 옆 힘 (왼쪽 타이어 기준). tire_model.formula_lateral 과 같되 밀림 항에 sign(γ) 가 없다
    (S_hy = p_hy1 + p_hy3·γ, S_vy = Fz·(p_vy1 + p_vy3·γ); 원본의 sign(γ)·|γ| 항은 γ 와 같으므로 캠버 비례 항은 같음)."""
    S_hy = p.p_hy1 + p.p_hy3 * gamma
    S_vy = F_z * (p.p_vy1 + p.p_vy3 * gamma)
    alpha_y = alpha + S_hy
    mu_y = p.p_dy1 * (1 - p.p_dy3 * gamma ** 2)
    C_y = p.p_cy1
    D_y = mu_y * F_z
    E_y = p.p_ey1
    B_y = F_z * p.p_ky1 / (C_y * D_y)
    return D_y * math.sin(C_y * math.atan(B_y * alpha_y - E_y * (B_y * alpha_y - math.atan(B_y * alpha_y)))) + S_vy, mu_y


def vehicle_dynamics_mb(x, uInit, p):
    """
    vehicleDynamics_mb - multi-body vehicle dynamics based on the DOT (department of transportation) vehicle dynamics
    reference point: center of mass

    Syntax:
        f = vehicleDynamics_mb(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param uInit: vehicle input vector
        :param p: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations

    Author: Matthias Althoff
    Written: 05-January-2017
    Last update: 17-December-2017
    Last revision: ---
    """

    #------------- BEGIN CODE --------------

    # set gravity constant
    g = 9.81  #[m/s^2]

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle
    #x6 = yaw rate

    #x7 = roll angle
    #x8 = roll rate
    #x9 = pitch angle
    #x10 = pitch rate
    #x11 = velocity in y-direction
    #x12 = z-position
    #x13 = velocity in z-direction

    #x14 = roll angle front
    #x15 = roll rate front
    #x16 = velocity in y-direction front
    #x17 = z-position front
    #x18 = velocity in z-direction front

    #x19 = roll angle rear
    #x20 = roll rate rear
    #x21 = velocity in y-direction rear
    #x22 = z-position rear
    #x23 = velocity in z-direction rear

    #x24 = left front wheel angular speed
    #x25 = right front wheel angular speed
    #x26 = left rear wheel angular speed
    #x27 = right rear wheel angular speed

    #x28 = delta_y_f
    #x29 = delta_y_r

    #u1 = steering angle velocity of front wheels
    #u2 = acceleration

    #consider steering constraints
    u = []
    u.append(steering_constraints(x[2], uInit[0], p.steering)) # different name u_init/u due to side effects of u
    #consider acceleration constraints
    u.append(acceleration_constraints(x[3], uInit[1], p.longitudinal)) # different name u_init/u due to side effects of u

    #compute slip angle at cg
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        beta = 0.
    else:
        beta = math.atan(x[10]/x[3]) 
    vel = math.sqrt(x[3]**2 + x[10]**2) 



    #vertical tire forces
    F_z_LF = (x[16] + p.R_w*(math.cos(x[13]) - 1) - 0.5*p.T_f*math.sin(x[13]))*p.K_zt 
    F_z_RF = (x[16] + p.R_w*(math.cos(x[13]) - 1) + 0.5*p.T_f*math.sin(x[13]))*p.K_zt 
    F_z_LR = (x[21] + p.R_w*(math.cos(x[18]) - 1) - 0.5*p.T_r*math.sin(x[18]))*p.K_zt 
    F_z_RR = (x[21] + p.R_w*(math.cos(x[18]) - 1) + 0.5*p.T_r*math.sin(x[18]))*p.K_zt 

    #obtain individual tire speeds
    u_w_lf = (x[3] + 0.5*p.T_f*x[5])*math.cos(x[2]) + (x[10] + p.a*x[5])*math.sin(x[2]) 
    u_w_rf = (x[3] - 0.5*p.T_f*x[5])*math.cos(x[2]) + (x[10] + p.a*x[5])*math.sin(x[2]) 
    u_w_lr = x[3] + 0.5*p.T_r*x[5] 
    u_w_rr = x[3] - 0.5*p.T_r*x[5] 

    #negative wheel spin forbidden
    if u_w_lf < 0.0:
       u_w_lf *= 0

    if u_w_rf < 0.0:
       u_w_rf *= 0

    if u_w_lr < 0.0:
       u_w_lr *= 0

    if u_w_rr < 0.0:
       u_w_rr *= 0
    #compute longitudinal slip
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        s_lf = 0.
        s_rf = 0.
        s_lr = 0.
        s_rr = 0.
    else:
        s_lf = 1 - p.R_w*x[23]/u_w_lf 
        s_rf = 1 - p.R_w*x[24]/u_w_rf 
        s_lr = 1 - p.R_w*x[25]/u_w_lr 
        s_rr = 1 - p.R_w*x[26]/u_w_rr 

    #lateral slip angles
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        alpha_LF = 0.
        alpha_RF = 0.
        alpha_LR = 0.
        alpha_RR = 0.
    else:
        alpha_LF = math.atan((x[10] + p.a*x[5] - x[14]*(p.R_w - x[16]))/(x[3] + 0.5*p.T_f*x[5])) - x[2] 
        alpha_RF = math.atan((x[10] + p.a*x[5] - x[14]*(p.R_w - x[16]))/(x[3] - 0.5*p.T_f*x[5])) - x[2] 
        alpha_LR = math.atan((x[10] - p.b*x[5] - x[19]*(p.R_w - x[21]))/(x[3] + 0.5*p.T_r*x[5])) 
        alpha_RR = math.atan((x[10] - p.b*x[5] - x[19]*(p.R_w - x[21]))/(x[3] - 0.5*p.T_r*x[5])) 

    #auxiliary suspension movement
    z_SLF = (p.h_s - p.R_w + x[16] - x[11])/math.cos(x[6]) - p.h_s + p.R_w + p.a*x[8] + 0.5*(x[6] - x[13])*p.T_f 
    z_SRF = (p.h_s - p.R_w + x[16] - x[11])/math.cos(x[6]) - p.h_s + p.R_w + p.a*x[8] - 0.5*(x[6] - x[13])*p.T_f 
    z_SLR = (p.h_s - p.R_w + x[21] - x[11])/math.cos(x[6]) - p.h_s + p.R_w - p.b*x[8] + 0.5*(x[6] - x[18])*p.T_r 
    z_SRR = (p.h_s - p.R_w + x[21] - x[11])/math.cos(x[6]) - p.h_s + p.R_w - p.b*x[8] - 0.5*(x[6] - x[18])*p.T_r 

    dz_SLF = x[17] - x[12] + p.a*x[9] + 0.5*(x[7] - x[14])*p.T_f 
    dz_SRF = x[17] - x[12] + p.a*x[9] - 0.5*(x[7] - x[14])*p.T_f 
    dz_SLR = x[22] - x[12] - p.b*x[9] + 0.5*(x[7] - x[19])*p.T_r 
    dz_SRR = x[22] - x[12] - p.b*x[9] - 0.5*(x[7] - x[19])*p.T_r 

    #camber angles
    gamma_LF = x[6] + p.D_f*z_SLF + p.E_f*(z_SLF)**2 
    gamma_RF = x[6] - p.D_f*z_SRF - p.E_f*(z_SRF)**2 
    gamma_LR = x[6] + p.D_r*z_SLR + p.E_r*(z_SLR)**2 
    gamma_RR = x[6] - p.D_r*z_SRR - p.E_r*(z_SRR)**2 

    #compute longitudinal tire forces using the magic formula for pure slip
    F0_x_LF = tireModel.formula_longitudinal(s_lf, gamma_LF, F_z_LF, p.tire_f)
    F0_x_RF = tireModel.formula_longitudinal(s_rf, gamma_RF, F_z_RF, p.tire_f)
    F0_x_LR = tireModel.formula_longitudinal(s_lr, gamma_LR, F_z_LR, p.tire_r)
    F0_x_RR = tireModel.formula_longitudinal(s_rr, gamma_RR, F_z_RR, p.tire_r)

    #compute lateral tire forces using the magic formula for pure slip
    if getattr(p, "tire_mirror", False):  # [11/21] 표준 식 + 오른쪽 거울상 (모듈 설명)
        F0_y_LF, mu_y_LF = _lateral_std(alpha_LF, gamma_LF, F_z_LF, p.tire_f)
        F0_y_RF, mu_y_RF = _lateral_std(-alpha_RF, -gamma_RF, F_z_RF, p.tire_f)  # 거울상 입력으로 계산한 값 (부호는 결합 뒤에 뒤집음)
        F0_y_LR, mu_y_LR = _lateral_std(alpha_LR, gamma_LR, F_z_LR, p.tire_r)
        F0_y_RR, mu_y_RR = _lateral_std(-alpha_RR, -gamma_RR, F_z_RR, p.tire_r)
    else:
        res = tireModel.formula_lateral(alpha_LF, gamma_LF, F_z_LF, p.tire_f)
        F0_y_LF = res[0]
        mu_y_LF = res[1]
        res = tireModel.formula_lateral(alpha_RF, gamma_RF, F_z_RF, p.tire_f)
        F0_y_RF = res[0]
        mu_y_RF = res[1]
        res = tireModel.formula_lateral(alpha_LR, gamma_LR, F_z_LR, p.tire_r)
        F0_y_LR = res[0]
        mu_y_LR = res[1]
        res = tireModel.formula_lateral(alpha_RR, gamma_RR, F_z_RR, p.tire_r)
        F0_y_RR = res[0]
        mu_y_RR = res[1]

    #compute longitudinal tire forces using the magic formula for combined slip
    F_x_LF = tireModel.formula_longitudinal_comb(s_lf, alpha_LF, F0_x_LF, p.tire_f)
    F_x_RF = tireModel.formula_longitudinal_comb(s_rf, alpha_RF, F0_x_RF, p.tire_f)
    F_x_LR = tireModel.formula_longitudinal_comb(s_lr, alpha_LR, F0_x_LR, p.tire_r)
    F_x_RR = tireModel.formula_longitudinal_comb(s_rr, alpha_RR, F0_x_RR, p.tire_r)

    #compute lateral tire forces using the magic formula for combined slip
    F_y_LF = tireModel.formula_lateral_comb(s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, p.tire_f)
    F_y_LR = tireModel.formula_lateral_comb(s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, p.tire_r)
    if getattr(p, "tire_mirror", False):  # 오른쪽: 거울상 입력으로 결합 미끄럼 계산 후 부호를 뒤집음
        F_y_RF = -tireModel.formula_lateral_comb(s_rf, -alpha_RF, -gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, p.tire_f)
        F_y_RR = -tireModel.formula_lateral_comb(s_rr, -alpha_RR, -gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, p.tire_r)
    else:
        F_y_RF = tireModel.formula_lateral_comb(s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, p.tire_f)
        F_y_RR = tireModel.formula_lateral_comb(s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, p.tire_r)

    #auxiliary movements for compliant joint equations
    delta_z_f = p.h_s - p.R_w + x[16] - x[11] 
    delta_z_r = p.h_s - p.R_w + x[21] - x[11] 

    delta_phi_f = x[6] - x[13] 
    delta_phi_r = x[6] - x[18] 

    dot_delta_phi_f = x[7] - x[14] 
    dot_delta_phi_r = x[7] - x[19] 

    dot_delta_z_f = x[17] - x[12] 
    dot_delta_z_r = x[22] - x[12] 

    dot_delta_y_f = x[10] + p.a*x[5] - x[15] 
    dot_delta_y_r = x[10] - p.b*x[5] - x[20] 

    delta_f = delta_z_f*math.sin(x[6]) - x[27]*math.cos(x[6]) - (p.h_raf - p.R_w)*math.sin(delta_phi_f) 
    delta_r = delta_z_r*math.sin(x[6]) - x[28]*math.cos(x[6]) - (p.h_rar - p.R_w)*math.sin(delta_phi_r) 

    dot_delta_f = (delta_z_f*math.cos(x[6]) + x[27]*math.sin(x[6]))*x[7] + dot_delta_z_f*math.sin(x[6]) - dot_delta_y_f*math.cos(x[6]) - (p.h_raf - p.R_w)*math.cos(delta_phi_f)*dot_delta_phi_f 
    dot_delta_r = (delta_z_r*math.cos(x[6]) + x[28]*math.sin(x[6]))*x[7] + dot_delta_z_r*math.sin(x[6]) - dot_delta_y_r*math.cos(x[6]) - (p.h_rar - p.R_w)*math.cos(delta_phi_r)*dot_delta_phi_r     
              
    #compliant joint forces
    F_RAF = delta_f*p.K_ras + dot_delta_f*p.K_rad 
    F_RAR = delta_r*p.K_ras + dot_delta_r*p.K_rad 

    #auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    F_SLF = p.m_s*g*p.b/(2*(p.a+p.b)) - z_SLF*p.K_sf - dz_SLF*p.K_sdf + (x[6] - x[13])*p.K_tsf/p.T_f 

    F_SRF = p.m_s*g*p.b/(2*(p.a+p.b)) - z_SRF*p.K_sf - dz_SRF*p.K_sdf - (x[6] - x[13])*p.K_tsf/p.T_f 

    F_SLR = p.m_s*g*p.a/(2*(p.a+p.b)) - z_SLR*p.K_sr - dz_SLR*p.K_sdr + (x[6] - x[18])*p.K_tsr/p.T_r 

    F_SRR = p.m_s*g*p.a/(2*(p.a+p.b)) - z_SRR*p.K_sr - dz_SRR*p.K_sdr - (x[6] - x[18])*p.K_tsr/p.T_r 


    #auxiliary variables sprung mass
    sumX = F_x_LR + F_x_RR + (F_x_LF + F_x_RF)*math.cos(x[2]) - (F_y_LF + F_y_RF)*math.sin(x[2]) 

    sumN = (F_y_LF + F_y_RF)*p.a*math.cos(x[2]) + (F_x_LF + F_x_RF)*p.a*math.sin(x[2]) \
           + (F_y_RF - F_y_LF)*0.5*p.T_f*math.sin(x[2]) + (F_x_LF - F_x_RF)*0.5*p.T_f*math.cos(x[2]) \
           + (F_x_LR - F_x_RR)*0.5*p.T_r - (F_y_LR + F_y_RR)*p.b 
       
    sumY_s = (F_RAF + F_RAR)*math.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR)*math.sin(x[6]) 

    sumL = 0.5*F_SLF*p.T_f + 0.5*F_SLR*p.T_r - 0.5*F_SRF*p.T_f - 0.5*F_SRR*p.T_r \
           - F_RAF/math.cos(x[6])*(p.h_s - x[11] - p.R_w + x[16] - (p.h_raf - p.R_w)*math.cos(x[13])) \
           - F_RAR/math.cos(x[6])*(p.h_s - x[11] - p.R_w + x[21] - (p.h_rar - p.R_w)*math.cos(x[18])) 
       
    sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR)*math.cos(x[6]) - (F_RAF + F_RAR)*math.sin(x[6]) 

    sumM_s = p.a*(F_SLF + F_SRF) - p.b*(F_SLR + F_SRR) + ((F_x_LF + F_x_RF)*math.cos(x[2]) \
           - (F_y_LF + F_y_RF)*math.sin(x[2]) + F_x_LR + F_x_RR)*(p.h_s - x[11]) 

    #auxiliary variables unsprung mass
    sumL_uf = 0.5*F_SRF*p.T_f - 0.5*F_SLF*p.T_f - F_RAF*(p.h_raf - p.R_w) \
              + F_z_LF*(p.R_w*math.sin(x[13]) + 0.5*p.T_f*math.cos(x[13]) - p.K_lt*F_y_LF) \
              - F_z_RF*(-p.R_w*math.sin(x[13]) + 0.5*p.T_f*math.cos(x[13]) + p.K_lt*F_y_RF) \
              - ((F_y_LF + F_y_RF)*math.cos(x[2]) + (F_x_LF + F_x_RF)*math.sin(x[2]))*(p.R_w - x[16]) 
          
    sumL_ur = 0.5*F_SRR*p.T_r - 0.5*F_SLR*p.T_r - F_RAR*(p.h_rar - p.R_w) \
              + F_z_LR*(p.R_w*math.sin(x[18]) + 0.5*p.T_r*math.cos(x[18]) - p.K_lt*F_y_LR) \
              - F_z_RR*(-p.R_w*math.sin(x[18]) + 0.5*p.T_r*math.cos(x[18]) + p.K_lt*F_y_RR) \
              - (F_y_LR + F_y_RR)*(p.R_w - x[21])    
          
    sumZ_uf = F_z_LF + F_z_RF + F_RAF*math.sin(x[6]) - (F_SLF + F_SRF)*math.cos(x[6]) 

    sumZ_ur = F_z_LR + F_z_RR + F_RAR*math.sin(x[6]) - (F_SLR + F_SRR)*math.cos(x[6]) 

    sumY_uf = (F_y_LF + F_y_RF)*math.cos(x[2]) + (F_x_LF + F_x_RF)*math.sin(x[2]) \
              - F_RAF*math.cos(x[6]) - (F_SLF + F_SRF)*math.sin(x[6]) 
          
    sumY_ur = (F_y_LR + F_y_RR) \
              - F_RAR*math.cos(x[6]) - (F_SLR + F_SRR)*math.sin(x[6])  
          
          
    #dynamics common with single-track model
    f = [] # init 'right hand side'
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        #wheelbase
        # lwb = p.a + p.b
        
        #system dynamics
        # x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        # f_ks = vehicle_dynamics_ks(x_ks, u, p)
        # f.extend(f_ks)
        # f.append(u[1]*lwb*math.tan(x[2]) + x[3]/(lwb*math.cos(x[2])**2)*u[0])

        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = p.a + p.b
        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        # kinematic model
        f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]]
        # derivative of slip angle and yaw rate
        d_beta = (p.b * u[0]) / (lwb * math.cos(x[2]) ** 2 * (1 + (math.tan(x[2]) ** 2 * p.b / lwb) ** 2))
        dd_psi = 1 / lwb * (u[1] * math.cos(x[6]) * math.tan(x[2]) -
                            x[3] * math.sin(x[6]) * d_beta * math.tan(x[2]) +
                            x[3] * math.cos(x[6]) * u[0] / math.cos(x[2]) ** 2)
        f.append(dd_psi)

    else:
        f.append(math.cos(beta + x[4])*vel)
        f.append(math.sin(beta + x[4])*vel)
        f.append(u[0])
        f.append(1/p.m*sumX + x[5]*x[10])
        f.append(x[5])
        f.append(1/(p.I_z - (p.I_xz_s)**2/p.I_Phi_s)*(sumN + p.I_xz_s/p.I_Phi_s*sumL))


    # remaining sprung mass dynamics
    f.append(x[7])
    f.append(1/(p.I_Phi_s - (p.I_xz_s)**2/p.I_z)*(p.I_xz_s/p.I_z*sumN + sumL))
    f.append(x[9])
    f.append(1/p.I_y_s*sumM_s)
    f.append(1/p.m_s*sumY_s - x[5]*x[3])
    f.append(x[12])
    f.append(g - 1/p.m_s*sumZ_s)

    #unsprung mass dynamics (front)
    f.append(x[14])
    f.append(1/p.I_uf*sumL_uf)
    f.append(1/p.m_uf*sumY_uf - x[5]*x[3])
    f.append(x[17])
    f.append(g - 1/p.m_uf*sumZ_uf)

    #unsprung mass dynamics (rear)
    f.append(x[19])
    f.append(1/p.I_ur*sumL_ur)
    f.append(1/p.m_ur*sumY_ur - x[5]*x[3])
    f.append(x[22])
    f.append(g - 1/p.m_ur*sumZ_ur)

    #convert acceleration input to brake and engine torque
    if u[1]>0:
        T_B = 0.0
        T_E = p.m*p.R_w*u[1] 
    else:
        T_B = p.m*p.R_w*u[1] 
        T_E = 0.



    # [Golf] ABS: 미끄럼률이 abs_slip 구간을 넘는 바퀴의 브레이크 토크를 줄임 (바퀴별 계수 k, 1 = 그대로)
    abs_slip = getattr(p, "abs_slip", None)
    if abs_slip and T_B < 0:
        k_abs = [1.0 - min(max((sl - abs_slip[0]) / (abs_slip[1] - abs_slip[0]), 0.0), 1.0) for sl in (s_lf, s_rf, s_lr, s_rr)]
    else:
        k_abs = [1.0, 1.0, 1.0, 1.0]

    # [11/22] 앞 차동장치 LSD (선택, p.lsd_mode). 마찰 클러치: 빠른 바퀴 → 느린 바퀴로 토크 dT 이동 (|dT| ≤ 잠김 토크).
    # 회전 차이 0 근처는 tanh 로 매끄럽게 (TUM Open-Car-Dynamics drivetrain_rwd_lsd 와 같은 형태, eps = 1 rad/s).
    # 'passive': 잠김 토크 = T0 + G·max(T_E, 0)  (기본 + 구동 토크 비율; TUM OCD, van Koutrik 2015 와 같은 형태)
    # 'active' : 잠김 토크 = u·Tmax  (VW VAQ: 유압 비례, 최대 1,600 Nm, u = 작동 비율 0~1, 밖에서 넣음)
    # None (기본): 열린 차동장치, 원본과 같음
    dT = 0.0
    lsd_mode = getattr(p, "lsd_mode", None)
    if lsd_mode:
        T_lock = (p.lsd_T0 + p.lsd_G * max(T_E, 0.0)) if lsd_mode == "passive" else p.lsd_u * p.lsd_Tmax
        dT = T_lock * math.tanh((x[23] - x[24]) / p.lsd_eps)  # 왼쪽이 빠르면 + : 왼쪽 → 오른쪽으로 옮김

    #wheel dynamics (p.T  new parameter for torque splitting)
    f.append(1/p.I_y_w*(-p.R_w*F_x_LF + 0.5*p.T_sb*T_B*k_abs[0] + 0.5*p.T_se*T_E - 0.5*dT))
    f.append(1/p.I_y_w*(-p.R_w*F_x_RF + 0.5*p.T_sb*T_B*k_abs[1] + 0.5*p.T_se*T_E + 0.5*dT))
    f.append(1/p.I_y_w*(-p.R_w*F_x_LR + 0.5*(1-p.T_sb)*T_B*k_abs[2] + 0.5*(1-p.T_se)*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_RR + 0.5*(1-p.T_sb)*T_B*k_abs[3] + 0.5*(1-p.T_se)*T_E))

    #negative wheel spin forbidden
    for iState in range(23, 27):
        if x[iState] < 0.0:
           x[iState] = 0.0
           f[iState] = 0.0

    #compliant joint equations
    f.append(dot_delta_y_f)
    f.append(dot_delta_y_r)

    return f


    #------------- END OF CODE --------------
