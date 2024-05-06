#ifndef RTW_HEADER_flyBall_h_
#define RTW_HEADER_flyBall_h_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "flyBall_types.h"
#include <cstring>

#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   (rtmGetTPtr((rtm))[0])
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

#ifndef ODE4_INTG
#define ODE4_INTG

struct ODE4_IntgData {
  real_T *y;
  real_T *f[4];
};

#endif

class flyBall final
{
 public:
  struct B_flyBall_T {
    real_T y;
    real_T vy_m;
    real_T vy;
    real_T reward;
    real_T ay_c;
    real_T ay;
  };

  struct X_flyBall_T {
    real_T y_CSTATE;
    real_T vy_CSTATE;
  };

  struct XDot_flyBall_T {
    real_T y_CSTATE;
    real_T vy_CSTATE;
  };

  struct XDis_flyBall_T {
    boolean_T y_CSTATE;
    boolean_T vy_CSTATE;
  };

  struct P_flyBall_T {
    real_T control;
    real_T target_t;
    real_T target_vy;
    real_T target_y;
  };

  struct RT_MODEL_flyBall_T {
    const char_T *errorStatus;
    RTWSolverInfo solverInfo;
    X_flyBall_T *contStates;
    int_T *periodicContStateIndices;
    real_T *periodicContStateRanges;
    real_T *derivs;
    boolean_T *contStateDisabled;
    boolean_T zCCacheNeedsReset;
    boolean_T derivCacheNeedsReset;
    boolean_T CTOutputIncnstWithState;
    real_T odeY[2];
    real_T odeF[4][2];
    ODE4_IntgData intgData;
    struct {
      int_T numContStates;
      int_T numPeriodicContStates;
      int_T numSampTimes;
    } Sizes;

    struct {
      uint32_T clockTick0;
      time_T stepSize0;
      uint32_T clockTick1;
      SimTimeStep simTimeStep;
      boolean_T stopRequestedFlag;
      time_T *t;
      time_T tArray[2];
    } Timing;
  };

  flyBall(flyBall const&) = delete;
  flyBall& operator= (flyBall const&) & = delete;
  flyBall(flyBall &&) = delete;
  flyBall& operator= (flyBall &&) = delete;
  flyBall::RT_MODEL_flyBall_T * getRTM();
  void initialize();
  void step();

  // get access to the private members t, y, vy, ay, reward, target_y, target_vy, target_t
  double get_t();
  double get_y();
  double get_vy();
  double get_ay();
  double get_reward();
  double get_target_y();
  double get_target_vy();
  double get_target_t();

  // set control
  void set_control(double control);

  static void terminate();
  flyBall();
  ~flyBall();
 private:
  B_flyBall_T flyBall_B;
  static P_flyBall_T flyBall_P;
  X_flyBall_T flyBall_X;
  void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si );
  void flyBall_derivatives();
  RT_MODEL_flyBall_T flyBall_M;
};

#endif

