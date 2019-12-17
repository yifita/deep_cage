#!/bin/bash

SMOOTH=50
ITERS=3
SMOOTH_LAST=200
PREFIX="smooth${SMOOTH}iter${ITERS}last${SMOOTH_LAST}"
orig=$1
output=$orig

# Merge duplicate vertices
# : <<'COMMENT_OUT_TO_DEDUP_VERTS'
mkdir -p $output/orig
mv $orig/*Sa.* $orig/*Sb.* $output/orig
FILES=`ls -1 $output/orig/*-Sa.* $output/orig/*-Sb.*`; for f in $FILES; do echo $f ; ext="${f##*.}"; b=`basename $f .$ext`; MeshFix --verbose --v-weld 0.00001 $f $output/${b}.obj ; done
# COMMENT_OUT_TO_DEDUP_VERTS

# Generate samples
# : <<'COMMENT_OUT_TO_GENERATE_SAMPLES'
mkdir -p $output/vert_pts $output/extra_pts $output/pts
FILES=`ls -1 $orig/*-Sa.obj $orig/*-Sb.obj`; for f in $FILES ; do echo $f ; b=`basename $f` ; MeshSample -v $f $output/vert_pts/${b%obj}pts ; done
FILES=`ls -1 $orig/*-Sa.obj $orig/*-Sb.obj`; for f in $FILES ; do echo $f ; b=`basename $f` ; MeshSample -n3000 -s3 $f $output/extra_pts/${b%obj}pts ; done
FILES=`ls -1 $output/vert_pts/*.pts`; for f in $FILES ; do echo $f ; b=`basename $f` ; cat $f $output/extra_pts/$b > ${output}/pts/$b ; done
# COMMENT_OUT_TO_GENERATE_SAMPLES

# Do registration
# : <<'COMMENT_OUT_TO_REGISTER'
FILES=`ls -1 ${output}/pts/*-Sa.pts`
for f in $FILES
do
  b=`basename ${f%-Sa.pts}`
  echo $b
  mkdir -p $output/reg/${PREFIX}/$b
  Register --smooth ${SMOOTH} --rounds ${ITERS} --smooth-last ${SMOOTH_LAST} "${output}/pts/${b}-Sa.pts" "${output}/pts/${b}-Sb.pts" "$output/reg/${PREFIX}/$b/offsets.pts"
done
# COMMENT_OUT_TO_REGISTER

# Construct deformed source objs
# : <<'COMMENT_OUT_TO_DEFORM_MESHES'
FILES=`find $output/reg/${PREFIX} -name '*deformed.pts'`
for f in $FILES
do
  echo $f
  d=`dirname $f`
  head -n $(( $(wc -l $f | awk '{print $1}') - 3000 )) $f | sed -E 's/[ ][^ ]+[ ][^ ]+[ ][^ ]+$//' > $d/deformed_verts.pts
  b=`basename $d`
  m="$output/${b}-Sa.obj"
  sed -E 's/^/v /' $d/deformed_verts.pts > $d/deformed.obj
  sed -E '/^v /d' $m >> $d/deformed.obj
  mv $d/deformed.obj $output/${b}-Sab.obj
done
# COMMENT_OUT_TO_DEFORM_MESHES

# # Render results
# # : <<'COMMENT_OUT_TO_RENDER_RESULTS'
# HTMLDIR="$output/reg/${PREFIX}/html"
# mkdir -p $HTMLDIR
# echo "<html><head><title>${PREFIX}</title></head><body><table>" > $HTMLDIR/index.html
# echo "<tr><th>Source</th><th>Target</th><th>ICP</th></tr>" >> $HTMLDIR/index.html
# FILES=`find $output/reg/${PREFIX} -name 'deformed.obj'`
# for f in $FILES
# do
#   echo $f
#   d=`dirname $f`
#   b=`basename $d`
#   ma="manif/${b}-Sa.obj"
#   mb="manif/${b}-Sb.obj"
#   RenderShape -0 -v --+ $ma $HTMLDIR/${b}-Sa.jpg 600 600
#   RenderShape -0 -v --+ $mb $HTMLDIR/${b}-Sb.jpg 600 600
#   RenderShape -0 -v --+ $f $HTMLDIR/${b}-Sab.jpg 600 600
#   echo "<tr>" >> $HTMLDIR/index.html
#   echo "  <td><img src=${b}-Sa.jpg width=400px></td>" >> $HTMLDIR/index.html
#   echo "  <td><img src=${b}-Sb.jpg width=400px></td>" >> $HTMLDIR/index.html
#   echo "  <td><img src=${b}-Sab.jpg width=400px></td>" >> $HTMLDIR/index.html
#   echo "</tr>" >> $HTMLDIR/index.html
# done
# echo "</table></body></html>" >> $HTMLDIR/index.html
# # COMMENT_OUT_TO_RENDER_RESULTS
