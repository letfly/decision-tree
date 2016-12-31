/**
 * \file gboost.cc
 * \brief booster implementations
 * \author Yufan Fu: letflykid@gmail.com
 */
// implementation of boosters go to here
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <climits>
#include "../utils/gboost_utils.h"
#include "gboost.h"
#include "gboost_gbmbase.h"
namespace gboost{
namespace booster{
  /**
   * \brief create a gradient booster, given type of booster
   * \param booster_type type of gradient booster, can be used to specify implements
   * \return the pointer to the gradient booster created
   */
  IBooster *CreateBooster(int booster_type) {
    // TODO
    return NULL;
  }
}
}
