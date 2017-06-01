SET(LRT_SOURCES
	Init # rtInit
	FrameBuffer.cxx # rtCreateFB, rtDestroyFB etc
	FrameBuffer/PBOFrameBuffer
	FrameBuffer/GLTextureFB
    # rtRenderFrame	
)

if(WITH_SKEPU)
   set (LRT_SOURCES ${LRT_SOURCES}  render_skepu_omp)
else()
   set (LRT_SOURCES ${LRT_SOURCES}  render)
endif()

#	RenderToTexture 
#	Init


ADD_LIBRARY(LRT STATIC ${LRT_SOURCES})
TARGET_LINK_LIBRARIES(LRT ${GL_LIBS} RTTL  ${GLUT_glut_LIBRARY} $ENV{LIBS} rt)
#OPENGL_LIBRARIES
INSTALL(TARGETS LRT LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)


ADD_EXECUTABLE(test_static_cube test/api_static_cube/static_cube)
TARGET_LINK_LIBRARIES(test_static_cube LRT)

