import os, sys
sys.path.insert(0, os.path.abspath(".."))
from opensky_api  import OpenSkyApi
from haversine  import haversine
import logger
from record_test_0902 import init_record


class AirPlane:
    def __init__(self, latitude, longitude, callsign, geo_altitude, on_ground, heading, to_location):
        self.latitude = latitude
        self.longitude = longitude
        self.callsign = callsign
        self.geo_altitude = geo_altitude
        self.on_ground = on_ground
        self.heading = heading
        self.to_location = to_location
        self.is_ignored = False

    def set_ignored (self, is_ignored):
        self.is_ignored = is_ignored

    def __repr__(self):
        return repr(self.callsign + " / " + str(self.to_location) + " / " + str(self.is_ignored))


def exec_get_airplane_sound(is_call):
    lst2 = []
    is_call_api = is_call
    center_location = (37.5216018, 126.837741)  # 살레시오
    while not is_call_api:
        try:
            api = OpenSkyApi('rxgp1', 'tla0420!@')
            states = api.get_states(bbox=(34.3900458847, 38.6122429469, 126.117397903, 129.468304478))  # In Korea
            # logger.debug(states)
            lst1 = []
            for s in states.states:
                airplane_location = (s.latitude, s.longitude)
                to_location = haversine(center_location, airplane_location)
                air_plane = AirPlane(latitude=s.latitude,
                                     longitude=s.longitude,
                                     callsign=s.callsign, geo_altitude=s.geo_altitude,
                                     on_ground=s.on_ground, heading=s.heading,
                                     to_location=to_location)
                # logger.debug(lst2)
                for obj2 in lst2:
                    if obj2.callsign == air_plane.callsign:
                        air_plane.set_ignored(obj2.is_ignored)

                if not s.on_ground:
                    if to_location < 3:  # 반경 3키로 이내
                        lst1.append(air_plane)
                        # logger.debug(airplane)

            lst1 = sorted(lst1, key=lambda x: x.to_location)
            for obj2 in lst2:
                for obj1 in lst1:
                    if obj2.callsign == obj1.callsign:
                        if obj1.to_location < obj2.to_location:
                            if not obj2.is_ignored:
                                obj1.set_ignored(True)
                                logger.debug("Trigged :" + obj1.callsign)
                                logger.debug("[who1 : " + str(obj1.to_location) + "]")
                                logger.debug("[who2 : " + str(obj2.to_location) + "]")
                                init_record("PI1", obj1.callsign.strip() + '.wav')

        except Exception as err:
            logger.error("error start ================================ ")
            logger.error(err)
            logger.error("end of error ================================ ")

        lst2 = lst1


if __name__ == '__main__':
    exec_get_airplane_sound()
