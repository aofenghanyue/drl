#include "flyBall.h"
#include "rtwtypes.h"
#include "flyBall_private.h"

void flyBall::rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  time_T t { rtsiGetT(si) };

  time_T tnew { rtsiGetSolverStopTime(si) };

  time_T h { rtsiGetStepSize(si) };

  real_T *x { rtsiGetContStates(si) };

  ODE4_IntgData *id { static_cast<ODE4_IntgData *>(rtsiGetSolverData(si)) };

  real_T *y { id->y };

  real_T *f0 { id->f[0] };

  real_T *f1 { id->f[1] };

  real_T *f2 { id->f[2] };

  real_T *f3 { id->f[3] };

  real_T temp;
  int_T i;
  int_T nXc { 2 };

  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));
  rtsiSetdX(si, f0);
  flyBall_derivatives();
  temp = 0.5 * h;
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (temp*f0[i]);
  }

  rtsiSetT(si, t + temp);
  rtsiSetdX(si, f1);
  this->step();
  flyBall_derivatives();
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (temp*f1[i]);
  }

  rtsiSetdX(si, f2);
  this->step();
  flyBall_derivatives();
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (h*f2[i]);
  }

  rtsiSetT(si, tnew);
  rtsiSetdX(si, f3);
  this->step();
  flyBall_derivatives();
  temp = h / 6.0;
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + temp*(f0[i] + 2.0*f1[i] + 2.0*f2[i] + f3[i]);
  }

  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

void flyBall::step()
{
  real_T Clock;
  boolean_T OR;
  if (rtmIsMajorTimeStep((&flyBall_M))) {
    rtsiSetSolverStopTime(&(&flyBall_M)->solverInfo,(((&flyBall_M)
      ->Timing.clockTick0+1)*(&flyBall_M)->Timing.stepSize0));
  }

  if (rtmIsMinorTimeStep((&flyBall_M))) {
    (&flyBall_M)->Timing.t[0] = rtsiGetT(&(&flyBall_M)->solverInfo);
  }

  Clock = (&flyBall_M)->Timing.t[0];
  OR = ((Clock >= 10.0) || (flyBall_X.y_CSTATE <= 0.0));
  if (rtmIsMajorTimeStep((&flyBall_M)) && OR) {
    rtmSetStopRequested((&flyBall_M), 1);
  }

  flyBall_B.vy_m = flyBall_X.vy_CSTATE;
  flyBall_B.vy = flyBall_B.vy_m;
  flyBall_B.y = flyBall_X.y_CSTATE;
  if (rtmIsMajorTimeStep((&flyBall_M))) {
    if (OR) {
      real_T rtb_Gain;
      real_T rtb_Square1;
      real_T rtb_Square2;
      rtb_Gain = flyBall_B.vy - flyBall_P.target_vy;
      rtb_Square2 = rtb_Gain * rtb_Gain;
      rtb_Gain = (flyBall_B.y - flyBall_P.target_y) * 0.33333333333333331;
      rtb_Square1 = rtb_Gain * rtb_Gain;
      rtb_Gain = (Clock - flyBall_P.target_t) * 10.0;
      flyBall_B.reward = ((rtb_Gain * rtb_Gain + rtb_Square1) + rtb_Square2) *
        -0.01 + 5.0;
    } else {
      flyBall_B.reward = -0.001;
    }

    flyBall_B.ay_c = flyBall_P.control + -9.8;
    flyBall_B.ay = flyBall_B.ay_c;
  }

  if (rtmIsMajorTimeStep((&flyBall_M))) {
    rt_ertODEUpdateContinuousStates(&(&flyBall_M)->solverInfo);
    ++(&flyBall_M)->Timing.clockTick0;
    (&flyBall_M)->Timing.t[0] = rtsiGetSolverStopTime(&(&flyBall_M)->solverInfo);

    {
      (&flyBall_M)->Timing.clockTick1++;
    }
  }
}

void flyBall::flyBall_derivatives()
{
  flyBall::XDot_flyBall_T *_rtXdot;
  _rtXdot = ((XDot_flyBall_T *) (&flyBall_M)->derivs);
  _rtXdot->y_CSTATE = flyBall_B.vy_m;
  _rtXdot->vy_CSTATE = flyBall_B.ay_c;
}

void flyBall::initialize()
{
  // 初始化运行终止标志
  (void) std::memset((static_cast<void *>(&flyBall_M)), 0,
                     sizeof(RT_MODEL_flyBall_T));
  {
    rtsiSetSimTimeStepPtr(&(&flyBall_M)->solverInfo, &(&flyBall_M)
                          ->Timing.simTimeStep);
    rtsiSetTPtr(&(&flyBall_M)->solverInfo, &rtmGetTPtr((&flyBall_M)));
    rtsiSetStepSizePtr(&(&flyBall_M)->solverInfo, &(&flyBall_M)
                       ->Timing.stepSize0);
    rtsiSetdXPtr(&(&flyBall_M)->solverInfo, &(&flyBall_M)->derivs);
    rtsiSetContStatesPtr(&(&flyBall_M)->solverInfo, (real_T **) &(&flyBall_M)
                         ->contStates);
    rtsiSetNumContStatesPtr(&(&flyBall_M)->solverInfo, &(&flyBall_M)
      ->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&(&flyBall_M)->solverInfo, &(&flyBall_M)
      ->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&(&flyBall_M)->solverInfo, &(&flyBall_M
      )->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&(&flyBall_M)->solverInfo, &(&flyBall_M)
      ->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&(&flyBall_M)->solverInfo, (&rtmGetErrorStatus
      ((&flyBall_M))));
    rtsiSetRTModelPtr(&(&flyBall_M)->solverInfo, (&flyBall_M));
  }

  rtsiSetSimTimeStep(&(&flyBall_M)->solverInfo, MAJOR_TIME_STEP);
  (&flyBall_M)->intgData.y = (&flyBall_M)->odeY;
  (&flyBall_M)->intgData.f[0] = (&flyBall_M)->odeF[0];
  (&flyBall_M)->intgData.f[1] = (&flyBall_M)->odeF[1];
  (&flyBall_M)->intgData.f[2] = (&flyBall_M)->odeF[2];
  (&flyBall_M)->intgData.f[3] = (&flyBall_M)->odeF[3];
  (&flyBall_M)->contStates = ((X_flyBall_T *) &flyBall_X);
  rtsiSetSolverData(&(&flyBall_M)->solverInfo, static_cast<void *>(&(&flyBall_M
    )->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&(&flyBall_M)->solverInfo, false);
  rtsiSetSolverName(&(&flyBall_M)->solverInfo,"ode4");
  rtmSetTPtr((&flyBall_M), &(&flyBall_M)->Timing.tArray[0]);
  (&flyBall_M)->Timing.stepSize0 = 0.1;
  flyBall_X.y_CSTATE = 100.0;
  flyBall_X.vy_CSTATE = 0.0;
  flyBall_B.y = flyBall_X.y_CSTATE;
  flyBall_B.vy_m = flyBall_X.vy_CSTATE;
  flyBall_B.vy = flyBall_B.vy_m;
  flyBall_B.reward = 0.0;
  flyBall_B.ay_c = -9.8;
}

void flyBall::terminate()
{
  
}

double flyBall::get_t()
{
  return flyBall_M.Timing.t[0];
}

double flyBall::get_y()
{
  return flyBall_B.y;
}

double flyBall::get_vy()
{
  return flyBall_B.vy;
}

double flyBall::get_ay()
{
  return flyBall_B.ay;
}

double flyBall::get_reward()
{
  return flyBall_B.reward;
}

double flyBall::get_target_y()
{
  return flyBall_P.target_y;
}

double flyBall::get_target_vy()
{
  return flyBall_P.target_vy;
}

double flyBall::get_target_t()
{
  return flyBall_P.target_t;
}

void flyBall::set_control(double control)
{
  flyBall_P.control = control;
}

flyBall::flyBall() :
  flyBall_B(),
  flyBall_X(),
  flyBall_M()
{
}

flyBall::~flyBall()
{
}

flyBall::RT_MODEL_flyBall_T * flyBall::getRTM()
{
  return (&flyBall_M);
}
