#ifndef LIGHTINGDLGBASE_H
#define LIGHTINGDLGBASE_H
#include "ui_lightingdlgbase4.h"
using namespace Qt;
using namespace Ui;
class lightingdlgbase : public Ui_Dialog
{ 
    Q_OBJECT

public:
    lightingdlgbase( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~lightingdlgbase();

};

#endif // LIGHTINGDLGBASE_H
