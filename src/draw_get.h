// Automatically generated by params.awk

bool _view_set_get (struct _view_set * p) {
  Params params[] = {
    {"ty", pfloat, &p->ty},
    {"tx", pfloat, &p->tx},
    {"fov", pfloat, &p->fov},
    {"quat", pfloat, p->quat, 4},
    {"sy", pfloat, &p->sy},
    {"sz", pfloat, &p->sz},
    {"sx", pfloat, &p->sx},
    {"height", punsigned, &p->height},
    {"samples", punsigned, &p->samples},
    {"width", punsigned, &p->width},
    {"bg", pfloat, p->bg, 3},
    {"phi", pfloat, &p->phi},
    {"psi", pfloat, &p->psi},
    {"theta", pfloat, &p->theta},
    {"relative", pbool, &p->relative},
    {"res", pfloat, &p->res},
    {"camera", pstring, &p->camera},
    {"p1y", pfloat, &p->p1y},
    {"p2x", pfloat, &p->p2x},
    {"p2y", pfloat, &p->p2y},
    {"p1x", pfloat, &p->p1x},
    {NULL}
  };
  return parse_params (params);
}

bool _translate_get (struct _translate * p) {
  Params params[] = {
    {"y", pfloat, &p->y},
    {"z", pfloat, &p->z},
    {"x", pfloat, &p->x},
    {NULL}
  };
  return parse_params (params);
}

bool _mirror_get (struct _mirror * p) {
  Params params[] = {
    {"n", pdouble, &p->n, 3},
    {"alpha", pdouble, &p->alpha},
    {NULL}
  };
  return parse_params (params);
}

bool _draw_vof_get (struct _draw_vof * p) {
  Params params[] = {
    {"c", pstring, &p->c},
    {"s", pstring, &p->s},
    {"edges", pbool, &p->edges},
    {"larger", pdouble, &p->larger},
    {"filled", pint, &p->filled},
    {"color", pstring, &p->color},
    {"max", pdouble, &p->max},
    {"spread", pdouble, &p->spread},
    {"min", pdouble, &p->min},
    {"linear", pbool, &p->linear},
    {"lc", pfloat, p->lc, 3},
    {"lw", pfloat, &p->lw},
    {"fc", pfloat, p->fc, 3},
    {NULL}
  };
  return parse_params (params);
}

bool _isoline_get (struct _isoline * p) {
  Params params[] = {
    {"phi", pstring, &p->phi},
    {"val", pdouble, &p->val},
    {"n", pint, &p->n},
    {"c", pstring, &p->c},
    {"s", pstring, &p->s},
    {"edges", pbool, &p->edges},
    {"larger", pdouble, &p->larger},
    {"filled", pint, &p->filled},
    {"color", pstring, &p->color},
    {"max", pdouble, &p->max},
    {"spread", pdouble, &p->spread},
    {"min", pdouble, &p->min},
    {"linear", pbool, &p->linear},
    {"lc", pfloat, p->lc, 3},
    {"lw", pfloat, &p->lw},
    {"fc", pfloat, p->fc, 3},
    {NULL}
  };
  return parse_params (params);
}

bool _cells_get (struct _cells * p) {
  Params params[] = {
    {"n", pdouble, &p->n, 3},
    {"alpha", pdouble, &p->alpha},
    {"lw", pfloat, &p->lw},
    {"lc", pfloat, p->lc, 3},
    {NULL}
  };
  return parse_params (params);
}

bool _vectors_get (struct _vectors * p) {
  Params params[] = {
    {"u", pstring, &p->u},
    {"scale", pdouble, &p->scale},
    {"lw", pfloat, &p->lw},
    {"lc", pfloat, p->lc, 3},
    {NULL}
  };
  return parse_params (params);
}

bool _squares_get (struct _squares * p) {
  Params params[] = {
    {"color", pstring, &p->color},
    {"max", pdouble, &p->max},
    {"spread", pdouble, &p->spread},
    {"min", pdouble, &p->min},
    {"linear", pbool, &p->linear},
    {"lc", pfloat, p->lc, 3},
    {"fc", pfloat, p->fc, 3},
    {"n", pdouble, &p->n, 3},
    {"alpha", pdouble, &p->alpha},
    {NULL}
  };
  return parse_params (params);
}

bool _box_get (struct _box * p) {
  Params params[] = {
    {"notics", pbool, &p->notics},
    {"lw", pfloat, &p->lw},
    {"lc", pfloat, p->lc, 3},
    {NULL}
  };
  return parse_params (params);
}

bool _isosurface_get (struct _isosurface * p) {
  Params params[] = {
    {"f", pstring, &p->f},
    {"v", pdouble, &p->v},
    {"color", pstring, &p->color},
    {"max", pdouble, &p->max},
    {"spread", pdouble, &p->spread},
    {"min", pdouble, &p->min},
    {"linear", pbool, &p->linear},
    {"lc", pfloat, p->lc, 3},
    {"fc", pfloat, p->fc, 3},
    {NULL}
  };
  return parse_params (params);
}

bool _travelling_get (struct _travelling * p) {
  Params params[] = {
    {"end", pdouble, &p->end},
    {"start", pdouble, &p->start},
    {"ty", pfloat, &p->ty},
    {"quat", pfloat, p->quat, 4},
    {"fov", pfloat, &p->fov},
    {"tx", pfloat, &p->tx},
    {NULL}
  };
  return parse_params (params);
}

bool _draw_string_get (struct _draw_string * p) {
  Params params[] = {
    {"str", pstring, &p->str},
    {"pos", pint, &p->pos},
    {"size", pfloat, &p->size},
    {"lw", pfloat, &p->lw},
    {"lc", pfloat, p->lc, 3},
    {NULL}
  };
  return parse_params (params);
}
